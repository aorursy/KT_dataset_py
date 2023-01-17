# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train_V2.csv")

df_test = pd.read_csv("../input/test_V2.csv")

print("Train data set :\n",df_Train.head())
print("Test data set :\n", df_Test.head())
df_train.info()
df_train.loc[df_train['winPlacePerc'].isna()==True, 'winPlacePerc']= 0.5
df_test['winPlacePerc']= 0.0
df_train['Type']= 'Train'
df_test['Type']= 'Test'
print(df_train.shape, df_test.shape)
df= pd.concat([df_train, df_test], ignore_index=True)
print(df.shape)
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
df= reduce_mem_usage(df)
df.columns
#df= df.set_index(['Id', 'groupId', 'matchId','Type'])

Y_Column = 'winPlacePerc'

ColumnList = ['assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
       'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']
for colname in ColumnList:
    df[colname+ '_1']= df[colname]/ (max(df[colname])- min(df[colname]))
    print(colname,'_1',' created')
df['totalDistance'] = df['rideDistance'] + df['walkDistance'] + df['swimDistance']
df['kills_assists'] = (df['kills'] + df['assists'])    
df['healthitems'] = df['heals'] + df['boosts']
df['kills_']= df['kills'] + df['longestKill']+ df['killStreaks']
df['killtypes']= df['headshotKills'] + df['roadKills']+ df['teamKills']+ df['vehicleDestroys']
df['Others']= df['damageDealt'] + df['DBNOs']+ df['revives'] + df['weaponsAcquired']
df['Points_'] = df['killPoints'] + df['winPoints']
df['Rank_Points']= df.groupby(by = ['matchId'])['Points_'].rank(ascending= False)
df['Rank_kills']= df.groupby(by = ['matchId'])['kills_'].rank(ascending= False)
df['Rank_killassist']= df.groupby(by = ['matchId'])['kills_assists'].rank(ascending= False)
df['Rank_GroupPoints']= df.groupby(by = ['groupId'])['Points_'].rank(ascending= False)
df['Rank_Groupkills']= df.groupby(by = ['groupId'])['kills_'].rank(ascending= False)
df['Rank_killassist']= df.groupby(by = ['groupId'])['kills_assists'].rank(ascending= False)

ColumnList = df.columns.tolist()
print(ColumnList)
ColumnList.remove('Id')
ColumnList.remove('groupId')
ColumnList.remove('matchId')
ColumnList.remove('Type')
ColumnList.remove('winPlacePerc')
ColumnList.remove('matchType')

df_Train = df[df['Type']== 'Train']
df_Test = df[df['Type']== 'Test']

df_Train= df_Train.set_index(['Id', 'groupId', 'matchId','Type'])
df_Test= df_Test.set_index(['Id', 'groupId', 'matchId','Type'])

X_Train = df_Train[ColumnList]
Y_Train = df_Train[Y_Column]

X_test = df_Test[ColumnList]

print(X_Train.shape, Y_Train.shape, X_test.shape)
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
X_Train.dtypes
%%time
# For tuning parameters
d_train1 = lgb.Dataset(X_Train, label=Y_Train.values)
params = {}
params['learning_rate'] = 0.09
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'
params['sub_feature'] = 0.8
params['num_leaves'] = 1000
params['min_data'] = 1
params['max_depth'] = 400
params['min_gain_to_split']= 0.0000001
clf1 = lgb.train(params, d_train1, 1000)
y_pred=clf1.predict(X_test)
print(y_pred)
#[0.2522337  0.82816766 0.52467091 ... 0.40437544 0.70938478 0.22089762]
#CPU times: user 11 s, sys: 432 ms, total: 11.4 s
#Wall time: 3 s
# Restore some columns
X_test = X_test.reset_index()
X_test["winPlacePerc"] = y_pred
X_test = X_test[['Id',"winPlacePerc"]]
df_test = df_test[['Id','groupId']]
df_test= df_test.merge(X_test, on = ['Id'])
df_test[['Id',"winPlacePerc"]].to_csv("submission.csv", index=False)