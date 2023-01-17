# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# Have also incldued an external dataset that provides the stats of the TOP PUBG playes


import os
print(os.listdir("../input/pubg-finish-placement-prediction"))
print(os.listdir("../input/pubgplayerstats"))

# Any results you write to the current directory are saved as output.
# def reload_df():
#     train_df=pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
#     test_df=pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
#     top_player_stats_df=pd.read_csv('../input/pubgplayerstats/PUBG_Player_Statistics.csv')
# return train_df,test_df,top_player_stats_df
train_df=pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
test_df=pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
top_player_stats_df=pd.read_csv('../input/pubgplayerstats/PUBG_Player_Statistics.csv')
# train_df.info()
top_player_solo_stats_df=top_player_stats_df.filter(regex='solo')
top_player_duo_stats_df=top_player_stats_df.filter(regex='duo')
top_player_squad_stats_df=top_player_stats_df.filter(regex='squad')

print(len(top_player_stats_df)
      ,len(top_player_solo_stats_df)
      ,len(top_player_duo_stats_df)
      ,len(top_player_squad_stats_df))
print(len(top_player_stats_df.columns)
      ,len(top_player_solo_stats_df.columns)
      ,len(top_player_duo_stats_df.columns)
      ,len(top_player_squad_stats_df.columns))
#player and tracker id are not included 
#but its not needed anyway as we are going to join this with every row anyway
top_player_stats_df.describe()
corr = top_player_stats_df.corr()
# print("Correlation Matrix")
# c1 = corr.abs().unstack()
# d1=c1.sort_values(ascending = False).drop_duplicates()
# print(d1.head(50))
# corr.info(verbose=True)
print(train_df.matchType.unique())
print(test_df.matchType.unique())

train_df[['matchType','Id']].groupby(['matchType']).count()
train_df[['matchType','groupId']].groupby(['matchType']).count()
train_df.nunique()
train_df.boosts.unique()
print(len(train_df))
train_df=train_df[train_df.winPlacePerc.notnull()]
print(len(train_df))
train_df.info()
train_df[train_df.matchType.str.contains('normal')].groupby(['matchType']).count()
match_type_ls=[]
for i in train_df.matchType:
    if 'solo' in i:
        match_type_ls.append('solo')
    elif 'duo' in i:
        match_type_ls.append('duo')
    else:
        match_type_ls.append('squad')

match_type_test_ls=[]
for i in test_df.matchType:
    if 'solo' in i:
        match_type_test_ls.append('solo')
    elif 'duo' in i:
        match_type_test_ls.append('duo')
    else:
        match_type_test_ls.append('squad')    
train_df['matchTypeReduced']=match_type_ls
test_df['matchTypeReduced']=match_type_test_ls

train_df[['matchType','matchTypeReduced']].head(3)
cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType','matchTypeReduced']
cols_to_fit = [col for col in train_df.columns if col not in cols_to_drop]
corr = train_df[cols_to_fit].corr()

plt.figure(figsize=(9,7))
sns.heatmap(
    corr,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    linecolor='white',
    linewidths=0.1,
    cmap="RdBu"
)
plt.show()
groupid_train_df=train_df.groupby(['groupId']).size().to_frame('players_in_team')
groupid_test_df=test_df.groupby(['groupId']).size().to_frame('players_in_team')
train_df = train_df.merge(groupid_train_df, how='left', on=['groupId'])
test_df = test_df.merge(groupid_test_df, how='left', on=['groupId'])

cols_to_ignore = ['Id', 'matchId','winPlacePerc']
cols_to_check=[i for i in train_df.columns if i not in cols_to_ignore]

numeric_cols_train_list = [cname for cname in cols_to_check if 
                train_df[cname].dtype in ['int64', 'float64']]
numeric_cols_test_list = [cname for cname in cols_to_check if 
                test_df[cname].dtype in ['int64', 'float64']]
low_cardinality_train_cols_ls = [cname for cname in cols_to_check if 
                                train_df[cname].nunique() < 20 and
                                train_df[cname].dtype == "object"]
low_cardinality_test_cols_ls = [cname for cname in cols_to_check if 
                                test_df[cname].nunique() < 20 and
                                test_df[cname].dtype == "object"]

scaler = MinMaxScaler()
train_df[numeric_cols_train_list]=scaler.fit_transform(train_df[numeric_cols_train_list])
test_df[numeric_cols_test_list]=scaler.transform(test_df[numeric_cols_test_list])
cols_with_missing_train_ls = [col for col in train_df.columns 
                    if train_df[col].isnull().any()]
cols_with_missing_test_ls = [col for col in train_df.columns 
                    if train_df[col].isnull().any()]
print("No. of columns with Null values :",len(cols_with_missing_train_ls))
print("No. of columns with Null values :",len(cols_with_missing_test_ls))
team_skill=['assists','revives','teamKills']
tomb_raider=['boosts','heals','weaponsAcquired']
terminator=['killPlace','killPoints','kills','killStreaks','longestKill','roadKills','damageDealt','DBNOs','headshotKills']
runner=['swimDistance','rideDistance','walkDistance']
everythingelse=['vehicleDestroys','matchDuration','maxPlace']


train_df['team_skill']=train_df['assists']+train_df['revives']+train_df['teamKills']
test_df['team_skill']=test_df['assists']+test_df['revives']+test_df['teamKills']

# train_df[['team_skill']+team_skill]
train_df['tomb_raider']=train_df['boosts']+train_df['heals']+train_df['weaponsAcquired']
test_df['tomb_raider']=test_df['boosts']+test_df['heals']+test_df['weaponsAcquired']

# train_df[['tomb_raider']+tomb_raider]
train_df['terminator']=train_df['killPlace']+train_df['killPoints']+train_df['kills']+train_df['killStreaks']+train_df['longestKill']+train_df['damageDealt']+train_df['DBNOs']+train_df['headshotKills']
test_df['terminator']=test_df['killPlace']+test_df['killPoints']+test_df['kills']+test_df['killStreaks']+test_df['longestKill']+test_df['damageDealt']+test_df['DBNOs']+test_df['headshotKills']

# train_df[['terminator']+terminator]
train_df['runner']=train_df['swimDistance']+train_df['rideDistance']+train_df['walkDistance']
test_df['runner']=test_df['swimDistance']+test_df['rideDistance']+test_df['walkDistance']

train_df['everythingelse']=train_df['vehicleDestroys']+train_df['matchDuration']+train_df['maxPlace']
test_df['everythingelse']=test_df['vehicleDestroys']+test_df['matchDuration']+test_df['maxPlace']

train_df['total_kickass_score']=train_df['team_skill']+train_df['tomb_raider']+train_df['terminator']+train_df['runner']+train_df['everythingelse']
test_df['total_kickass_score']=test_df['team_skill']+test_df['tomb_raider']+test_df['terminator']+test_df['runner']+test_df['everythingelse']

# groupid_train_df=train_df.groupby(['groupId'])['total_kickass_score'].size().to_frame('players_in_team')
# groupid_test_df=test_df.groupby(['groupId'])['total_kickass_score'].size().to_frame('players_in_team')
# train_df = train_df.merge(groupid_train_df, how='left', on=['groupId'])
# test_df = test_df.merge(groupid_test_df, how='left', on=['groupId'])
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
lb_make_2 = LabelEncoder()
#train_1_df = pd.get_dummies(train_df, columns=['matchTypeReduced','matchType'],prefix=['encoded'])
train_df['matchTypeReduced_num'] = lb_make.fit_transform(train_df['matchTypeReduced'])
train_df['matchType_num'] = lb_make_2.fit_transform(train_df['matchType'])
test_df['matchTypeReduced_num']=lb_make.transform(test_df['matchTypeReduced'])
test_df['matchType_num']= lb_make_2.transform(test_df['matchType'])

columns_converted_to_int_ls=['matchTypeReduced_num','matchType_num']

my_train_cols = columns_converted_to_int_ls + numeric_cols_train_list+['winPlacePerc']
my_test_cols = columns_converted_to_int_ls + numeric_cols_test_list
train_x=train_df[my_train_cols]
train_y=train_df['winPlacePerc']
test_x=test_df[my_test_cols]
train_df_bkup=train_df
test_df_bkup=test_df
# train_df.drop(['groupId'],axis=1,inplace=True)
# test_df.drop(['groupId'],axis=1,inplace=True)
# print (len(train_df[(train_df.matchTypeReduced=='squad')&(train_df.players_in_team>4)].Id))
# print (len(train_df))
corr = train_df[['walkDistance', 'players_in_team','total_kickass_score', 'winPlacePerc']].corr()
sns.heatmap(
    corr,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    linecolor='white',
    linewidths=0.1,
    cmap="RdBu"
)
plt.show()
train_selected_df=train_df[['team_skill','terminator','runner','tomb_raider','everythingelse']+['players_in_team','winPlacePerc','matchTypeReduced_num','matchType_num','total_kickass_score']+terminator+runner+team_skill+tomb_raider+everythingelse]
test_selected_df=test_df[['team_skill','terminator','runner','tomb_raider','everythingelse']+['players_in_team','matchTypeReduced_num','matchType_num','total_kickass_score']+terminator+runner+team_skill+tomb_raider+everythingelse]
train_selected_x_df=train_selected_df[['team_skill','terminator','runner','tomb_raider','everythingelse']+['players_in_team','matchTypeReduced_num','matchType_num','total_kickass_score']+terminator+runner+team_skill+tomb_raider+everythingelse]
train_selected_y_df=train_selected_df[['winPlacePerc']]
test_selected_x_df=test_selected_df[['team_skill','terminator','runner','tomb_raider','everythingelse']+['players_in_team','matchTypeReduced_num','matchType_num','total_kickass_score']+terminator+runner+team_skill+tomb_raider+everythingelse]
print("The count of training dataset features are : ",len(train_selected_x_df))
print("The count of training dataset target variables are : ",len(train_selected_y_df))
print("The count of test dataset target features are : ",len(test_selected_x_df))
# corr=train_selected_df.corr()
# sns.heatmap(
#     corr,
#     xticklabels=corr.columns.values,
#     yticklabels=corr.columns.values,
#     linecolor='white',
#     linewidths=0.1,
#     cmap="RdBu"
# )
# # plt.show()
from xgboost import XGBRegressor

my_model = XGBRegressor(max_depth = 5 ,min_child_weight = 1,subsample=0.8,colsample_bytree = 0.8 ,scale_pos_weight = 1)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_selected_x_df, train_selected_y_df, verbose=True)
# #Import libraries:
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from xgboost.sklearn import XGBClassifier
# from sklearn import  metrics   #Additional scklearn functions
# #from sklearn.grid_search import GridSearchCV   #Perforing grid search

# def modelfit(alg, train_x,train_y,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
#     if useTrainCV:
#         xgb_param = alg.get_xgb_params()
#         xgtrain = xgb.DMatrix(train_x.values, label=train_y.values)
#         cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
#             metrics='mae', early_stopping_rounds=early_stopping_rounds, show_progress=False)
#         alg.set_params(n_estimators=cvresult.shape[0])
    
#     #Fit the algorithm on the data
#     alg.fit(train_x, train_y,eval_metric='mae')
        
#     #Predict training set:
#     dtrain_predictions = alg.predict(train_x)
#     dtrain_predprob = alg.predict_proba(train_x)[:,1]
        
#     #Print model report:
#     print ("\nModel Report")
#     #print "Accuracy : %.4g" % metrics.accuracy_score(train_y.values, dtrain_predictions)
#     print ("AUC Score (Train): %f") % metrics.mean_absolute_error(train_y, dtrain_predprob)
                    
#     feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#     feat_imp.plot(kind='bar', title='Feature Importances')
#     plt.ylabel('Feature Importance Score')
#     alg.predict(test_selected_x_df)
#     predictions_final=[0 if i < 0  else i for i in predictions]
#     my_3submission = pd.DataFrame({'Id': test_df.Id, 'winPlacePerc': predictions_final})
#     my_3submission.head()
#     os.chdir("/kaggle/working/")
#     my_3submission.to_csv('third_submit.csv',header=True, index=False)
# #Choose all predictors except target & IDcols
# #predictors = [x for x in train.columns if x not in [target, IDcol]]
# from xgboost import XGBRegressor
# xgb1 = XGBRegressor(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# modelfit(xgb1, train_selected_x_df,train_selected_y_df)
predictions = my_model.predict(test_selected_x_df)
predictions_final=[0 if i < 0  else i for i in predictions]
my_3submission = pd.DataFrame({'Id': test_df.Id, 'winPlacePerc': predictions_final})
my_3submission.head()
os.chdir("/kaggle/working/")
my_3submission.to_csv('third_submit.csv',header=True, index=False)
