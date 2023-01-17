import pandas as pd

import datetime

import lightgbm as lgb

import numpy as np

from sklearn import preprocessing
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

sub = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
train
train = train.append(test[test['Date']>'2020-03-31'])
train['Date'] = pd.to_datetime(train['Date'], format='%Y-%m-%d')
train['day_dist'] = train['Date']-train['Date'].min()
train['day_dist'] = train['day_dist'].dt.days
print(train['Date'].max())

#print(val['Date'].max())

print(test['Date'].min())

print(test['Date'].max())

#print(test['Date'].max()-test['Date'].min())
cat_cols = train.dtypes[train.dtypes=='object'].keys()

cat_cols
for cat_col in cat_cols:

    train[cat_col].fillna('no_value', inplace = True)
train['place'] = train['Province_State']+'_'+train['Country_Region']

#vcheck = train[(train['Date']>='2020-03-12')]
cat_cols = train.dtypes[train.dtypes=='object'].keys()

cat_cols
for cat_col in ['place']:

    #train[cat_col].fillna('no_value', inplace = True) #train[cat_col].value_counts().idxmax()

    le = preprocessing.LabelEncoder()

    le.fit(train[cat_col])

    train[cat_col]=le.transform(train[cat_col])
train
train.keys()
drop_cols = ['Id','ForecastId', 'ConfirmedCases','Date', 'Fatalities',

             'day_dist', 'Province_State', 'Country_Region'] #,'day_dist','shift_22_ft','shift_23_ft','shift_24_ft','shift_25_ft','shift_26_ft']
#val = train[(train['Id']).isnull()==True]

#train = train[(train['Id']).isnull()==False]

val = train[(train['Date']>='2020-03-12')&(train['Id'].isnull()==False)]

#test = train[(train['Date']>='2020-03-12')&(train['Id'].isnull()==True)]

#train = train[(train['Date']<'2020-03-22')&(train['Id'].isnull()==False)]
val
y_ft = train["Fatalities"]

y_val_ft = val["Fatalities"]







y_cc = train["ConfirmedCases"]

y_val_cc = val["ConfirmedCases"]



#train.drop(drop_cols, axis=1, inplace=True)

#test.drop(drop_cols, axis=1, inplace=True)

#val.drop(drop_cols, axis=1, inplace=True)
#损失函数

def rmsle (y_true, y_pred):

    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))
def mape (y_true, y_pred):

    return np.mean(np.abs(y_pred -y_true)*100/(y_true+1))
dates = test['Date'].unique()
dates = dates[dates>'2020-03-31']
len(dates)
params = {

    "objective": "regression",

    "boosting": 'gbdt', #"gbdt",

    "num_leaves": 1280,

    "learning_rate": 0.05,

    "feature_fraction": 0.9, # 0.9,

    "reg_lambda": 2,

    "metric": "rmse",

    'min_data_in_leaf':20

}
i=0 

fold_n =0

for date in dates:

    fold_n +=1

    i+=1

    if i ==1:

        nrounds = 200

    else:

        nrounds =100

    print(i)

    print(nrounds)

    #shift() 方法用于把数组的第一个元素从其中删除,并返回第一个元素的值。

    train['shift_1_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i)

    train['shift_2_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+1)

    train['shift_3_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+2)

    train['shift_4_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+3)

    train['shift_5_cc'] = train.groupby(['place'])['ConfirmedCases'].shift(i+4)

    

    val2 =train[train['Date']==date]

    train2 = train[(train['Date']<date)]

    y_cc = train2['ConfirmedCases']

    

    train2.drop(drop_cols, axis=1, inplace=True)

    val2.drop(drop_cols, axis=1, inplace=True)

    

    dtrain =lgb.Dataset(train2,label=y_cc)

    dvalid = lgb.Dataset(val2,label=y_val_cc)

    

    model = lgb.train(params,dtrain,nrounds,

                      categorical_feature=['place'],

                      verbose_eval=False)

    

    y_pred = model.predict(val2,num_iteration=nrounds)

    

    test.loc[test['Date']==date,'ConfirmedCases']=y_pred

    train.loc[train['Date']==date,'ConfirmedCases']=y_pred

    

    
train[train['Date']==date]
test[test['Country_Region']=='Italy']
test[(test['Country_Region']=='China')&(test['Province_State']=='Zhejiang')]
y_pred.mean()
i=0 

fold_n =0

for date in dates:

    fold_n +=1

    i+=1

    if i ==1:

        nrounds = 200

    else:

        nrounds =100

    print(i)

    print(nrounds)

    #shift() 方法用于把数组的第一个元素从其中删除,并返回第一个元素的值。

    train['shift_1_cc'] = train.groupby(['place'])['Fatalities'].shift(i)

    train['shift_2_cc'] = train.groupby(['place'])['Fatalities'].shift(i+1)

    train['shift_3_cc'] = train.groupby(['place'])['Fatalities'].shift(i+2)

    train['shift_4_cc'] = train.groupby(['place'])['Fatalities'].shift(i+3)

    train['shift_5_cc'] = train.groupby(['place'])['Fatalities'].shift(i+4)

    

    val2 =train[train['Date']==date]

    train2 = train[(train['Date']<date)]

    y_ft = train2['Fatalities']

    

    train2.drop(drop_cols, axis=1, inplace=True)

    val2.drop(drop_cols, axis=1, inplace=True)

    

    dtrain =lgb.Dataset(train2,label=y_ft)

    dvalid = lgb.Dataset(val2,label=y_val_ft)

    

    model = lgb.train(params,dtrain,nrounds,

                      categorical_feature=['place'],

                      verbose_eval=False)

    

    y_pred = model.predict(val2,num_iteration=nrounds)

    

    test.loc[test['Date']==date,'Fatalities']=y_pred

    train.loc[train['Date']==date,'Fatalities']=y_pred
test[test['Country_Region']=='Italy']
print(len(test))
train_sub = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
test = pd.merge(test,train_sub[['Province_State','Country_Region',

                                'Date','ConfirmedCases',

                                'Fatalities']],

                on=['Province_State','Country_Region',

                    'Date'], how='left')
print(len(test))
test.head()
test.loc[test['ConfirmedCases_x'].isnull()==True]
test.loc[test['ConfirmedCases_x'].isnull()==True, 'ConfirmedCases_x'] =test.loc[test['ConfirmedCases_x'].isnull()==True, 'ConfirmedCases_y']
test.loc[test['Fatalities_x'].isnull()==True, 'Fatalities_x'] = test.loc[test['Fatalities_x'].isnull()==True, 'Fatalities_y']
dates
last_amount = test.loc[(test['Country_Region']=='Italy')&(test['Date']=='2020-03-31'),'ConfirmedCases_x']
last_fat = test.loc[(test['Country_Region']=='Italy')&(test['Date']=='2020-03-31'),'Fatalities_x']
last_fat.values[0]
len(dates)
i = 0

k = 30
test.loc[(test['Country_Region']=='Italy')] #&(test['Date']==date),'ConfirmedCases_x' 
for date in dates:

    k = k-1

    i = i+1

    test.loc[(test['Country_Region']=='Italy')&(test['Date']==date),

            'ConfirmedCases_x']=last_amount.values[0] + i*(5000-(100*i))

    test.loc[(test['Country_Region']=='Italy')&(test['Date']==date),

             'Fatalities_x'] =  last_fat.values[0]+i*(800-(10*i))
test.loc[(test['Country_Region']=='Italy')] #&(test['Date']==date),'ConfirmedCases_x' 
last_amount = test.loc[(test['Country_Region']=='China')&(test['Province_State']!='Hubei')&(test['Date']=='2020-03-31'),'ConfirmedCases_x']

last_fat = test.loc[(test['Country_Region']=='China')&(test['Province_State']!='Hubei')&(test['Date']=='2020-03-31'),'Fatalities_x']
i = 0

k = 30

for date in dates:

    k = k-1

    i = i+1

    test.loc[(test['Country_Region']=='China')&(test['Province_State']!='Hubei')&(test['Date']==date),

             'Fatalities_x']= last_fat.values

    test.loc[(test['Country_Region']=='China')&(test['Province_State']!='Hubei')&(test['Date']==date),

             'ConfirmedCases_x']= last_amount.values + i
last_amount = test.loc[(test['Country_Region']=='China')&(test['Province_State']=='Hubei')&(test['Date']=='2020-03-31'),'ConfirmedCases_x']

last_fat = test.loc[(test['Country_Region']=='China')&(test['Province_State']=='Hubei')&(test['Date']=='2020-03-31'),'Fatalities_x']
k=30

i=0

for date in dates:

    k = k-1

    i = i+1

    test.loc[(test['Country_Region']=='China')&(test['Province_State']=='Hubei')&(test['Date']==date),'ConfirmedCases_x']= last_amount.values[0]

    test.loc[(test['Country_Region']=='China')&(test['Province_State']=='Hubei')&(test['Date']==date),'Fatalities_x']= last_fat.values[0] + i 
sub = test[['ForecastId','ConfirmedCases_x','Fatalities_x']]
sub
sub.columns=['ForecastId','ConfirmedCases','Fatalities']
sub.loc[sub['ConfirmedCases']<0,'ConfirmedCases']=0
sub.loc[sub['Fatalities']<0, 'Fatalities']=0
sub['Fatalities'].describe()
sub['ConfirmedCases'].describe()
sub.to_csv('submission.csv',index=False)