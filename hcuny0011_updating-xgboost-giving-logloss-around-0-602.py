import csv

import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier



from sklearn.cross_validation import train_test_split, cross_val_score
df_all=pd.read_csv('../input/data.csv')

df_teid=pd.read_csv('../input/sample_submission.csv')
#split data into training and testing

test_id=df_teid['shot_id'].tolist()
df_all['shot_made_flag'].value_counts()
df_all['combined_shot_type'].value_counts()
df_all['period'].value_counts()
df_all['shot_zone_area'].value_counts()
df_all['shot_zone_basic'].value_counts()
df_all['shot_type'].value_counts()
df_all['opponent'].value_counts()
#string format into datetime

df_all['game_date'] = pd.to_datetime(df_all['game_date'])
#if it is a back to back game 

#(df_all['game_date'][100]-df_all['game_date'][1]).days #int

def btb(lst): #0,1

    btb_lst=[0]

    flag=0

    for i in range(1,len(lst)):

        if (lst[i]-lst[i-1]).days==1:

            btb_lst.append(1)

            flag=1

        elif (lst[i]-lst[i-1]).days==0:

            btb_lst.append(flag)

        else:

            flag=0

            btb_lst.append(flag)

    return btb_lst



df_all['btb']=btb(df_all['game_date'])
#home court game or not

def homecourt(row):

    if '@' in row:

        return 0

    elif 'vs' in row:

        return 1

    else:

        return 'error'

    





homecourt_label=df_all['matchup'].apply(homecourt)

df_all['homecourt']=homecourt_label
#game month

#w['female'] = w['female'].map({'female': 1, 'male': 0})

def gamemonth(lst):

    dict_month={'1': 'Jan', '2': 'Feb', '3': 'Mar',

                '4': 'Apr', '5': 'May', '6': 'Jun',

                '7': 'Jul', '8': 'Aug', '9': 'Sep',

                '10': 'Oct', '11': 'Nov', '12': 'Dec'}

    

    splitseries=lst.apply(lambda row :str(row.month))

    

    newseries=splitseries.map(dict_month)

    

    return newseries

    

gamemonth(df_all['game_date']).isnull().values.any() #false

df_all['gamemonth']=gamemonth(df_all['game_date'])
#if last shot was made

def lastshot(lst):

    last=[0]

    for i in range(1,len(lst)):

        

        if lst[i-1]==0:

            flag=0

            last.append(0)

        elif lst[i-1]==1:

            flag=1

            last.append(1)

        else:

            last.append('unknown') #due to the random test data

    return last



df_all['last_shot_flag']=lastshot(df_all['shot_made_flag'])
#add column secondsToPeriodEnd

df_all['secondsToPeriodEnd'] = 60*df_all['minutes_remaining']+df_all['seconds_remaining']
#add column secondsFromPeriodEnd

df_all['secondsFromGameStart'] = df_all['period'].astype(int)*12*60 - df_all['secondsToPeriodEnd']
criterion = df_all['shot_id'].map(lambda x: x not in test_id)

criterion1 = df_all['shot_id'].map(lambda x: x in test_id)

df_all_tr=df_all[criterion]

df_all_te=df_all[criterion1]
ctg_feature=['combined_shot_type', 

             'shot_id','shot_type', 

             'action_type',

             'shot_zone_area', 'shot_zone_basic', 'playoffs', 'period','opponent','season',

             'homecourt',

             'btb',

             'gamemonth',

            'last_shot_flag'

             ]

num_feature=['loc_x', 'loc_y', 'shot_distance','shot_id','seconds_remaining',

             'secondsToPeriodEnd','secondsFromGameStart']
df_ctg = df_all.loc[:, lambda df: ctg_feature]

encoded_ctg=pd.get_dummies(df_ctg).astype(np.int16)
criterion01 = encoded_ctg['shot_id'].map(lambda x: x not in test_id)

criterion11 = encoded_ctg['shot_id'].map(lambda x: x in test_id)





df_tr_ctg=encoded_ctg[criterion01]

df_te_ctg=encoded_ctg[criterion11]


df_tr_num = df_all_tr.loc[:, lambda df: num_feature]

df_te_num = df_all_te.loc[:, lambda df: num_feature]

flag = df_all_tr['shot_made_flag']


train=pd.merge(df_tr_ctg, df_tr_num,on='shot_id')



test=pd.merge(df_te_ctg, df_te_num,on='shot_id')
train.shape
#how new features look like

df_all.loc[:, lambda df: ['homecourt','btb','gamemonth','last_shot_flag']]
X_dtrain, X_deval, y_dtrain, y_deval = train_test_split(train, flag, random_state=2046, test_size=0.15)

prior = 0.4

dtrain = xgb.DMatrix(X_dtrain, y_dtrain)

deval = xgb.DMatrix(X_deval, y_deval)

watchlist = [(deval, 'eval')]

params = {

    'booster': 'gbtree',

    'objective': 'binary:logistic',

    'colsample_bytree': 0.8,

    'eta': 0.1,

    'max_depth': 3,

    'seed': 2017,

    'silent': 1,

   # 'gamma':0.005,

    'subsample':0.8,

     'base_score': prior,

    'eval_metric': 'logloss'

}



clf = xgb.train(params, dtrain, 200, watchlist, early_stopping_rounds=50)
pred=clf.predict(xgb.DMatrix(test))
lstY1103=pred.tolist()
lstY1103_1=['shot_made_flag']+lstY1103

lstID=['shot_id']+test_id
lstY1103_1