# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Thanks and credited to https://www.kaggle.com/gemartin who created this wonderful mem reducer
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df
print('-' * 80)
print('train')
train = import_data('../input/train.csv')

print('-' * 80)
print('test')
test = import_data('../input/test.csv')

print('-' * 80)
print('sample_submission')
submission = import_data('../input/sample_submission.csv')
train.info()
train.head()
train.isnull().sum()
y = train['winPlacePerc']
columns_todrop = ['winPlacePerc']
train_sel = train.drop(columns_todrop,axis=1)
train_mean = train_sel.groupby(['matchId','groupId']).mean()
test_mean = test.groupby(['matchId','groupId']).mean()

train_median = train_sel.groupby(['matchId','groupId']).median()
test_median = test.groupby(['matchId','groupId']).median()

train_rank = train_mean.groupby('matchId').rank(pct=True)
test_rank = test_mean.groupby('matchId').rank(pct=True)

train_mean.head()
train_rank.head()
train_new = pd.merge(train_sel,train_mean,suffixes=['',"_mean"],how="left",on=['matchId','groupId'])
test_new = pd.merge(test,test_mean,suffixes=['',"_mean"],how="left",on=['matchId','groupId'])

train_new = pd.merge(train_new,train_rank,suffixes=['',"_rank"],how="left",on=['matchId','groupId'])
test_new = pd.merge(test_new,test_rank,suffixes=['',"_rank"],how="left",on=['matchId','groupId'])

train_new = pd.merge(train_new,train_median,suffixes=['',"_median"],how="left",on=['matchId','groupId'])
test_new = pd.merge(test_new,test_median,suffixes=['',"_median"],how="left",on=['matchId','groupId'])

del train_mean
del test_mean

del train_rank
del test_rank


del train_median
del test_median
selected_columns=[]
for each in train_new.columns:
    if "_" in each:
        selected_columns.append(each)
train_selected = train_new[selected_columns]
test_selected = test_new[selected_columns]

train_selected['matchId'] = train['matchId']
test_selected['matchId'] = test['matchId']
print(selected_columns)
correlation = train_selected.corr()
plt.figure(figsize=(15,20))
sns.heatmap(correlation,xticklabels=train_selected.columns.values,yticklabels=train_selected.columns.values,cmap="PiYG")
#cols_toDrop = ['killPoints_mean','numGroups_mean','maxPlace_mean','vehicleDestroys_mean','Id_rank','killPoints_rank','maxPlace_rank','numGroups_rank','roadKills_rank','vehicleDestroys_rank']
cols_toDrop = ['Id_rank','Id_mean','Id_median','vehicleDestroys_mean','vehicleDestroys_median','vehicleDestroys_mean']
train_selected.drop(cols_toDrop,axis=1,inplace=True)
test_selected.drop(cols_toDrop,axis=1,inplace=True)
from sklearn.model_selection import train_test_split

train_selected['winPlacePerc'] = y
matchId = train_selected['matchId'].unique()
matchIdTrain = np.random.choice(matchId, int(0.80*len(matchId)))

df_train2 = df_train[train_selected['matchId'].isin(matchIdTrain)]
df_test = df_train[~train_selected['matchId'].isin(matchIdTrain)]

y_train = df_train2['winPlacePerc']
X_train = df_train2.drop(columns=['winPlacePerc'])
y_test = df_test['winPlacePerc']
X_test = df_test.drop(columns=['winPlacePerc'])
#X_train,X_test,y_train,y_test = train_test_split(train_selected,y,test_size=0.33)
X_train.shape
y_train.shape
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()#XGBRegressor(n_estimators=1000, learning_rate=0.05)
#pca = PCA(n_components=10)
regr = make_pipeline(model)
model.fit(X_train,y_train)#,early_stopping_rounds=5,eval_set=[(X_test, y_test)], verbose=False)


y_pred = model.predict(X_test)

print(mean_absolute_error(y_pred,y_test))
#columns_todrop = ['Id','groupId','matchId']
#test_sel = test.drop(columns_todrop,axis=1)
test_scaled = scaler.transform(test_selected)
final_pred = model.predict(test_scaled)
print(final_pred)
finalPred_series = pd.Series(final_pred)
submission = pd.concat([test['Id'],finalPred_series],axis=1)
columns=['Id','winPlacePerc']
submission.columns = columns
print(submission)
submission.to_csv('submission.csv', index=False)
