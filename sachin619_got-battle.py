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
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

    if verbose: print('Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train_df=reduce_mem_usage(pd.read_csv("../input/got-battle-of-blackwater/train.csv"))

test_df=reduce_mem_usage(pd.read_csv("../input/got-battle-of-blackwater/test.csv"))

pub1=reduce_mem_usage(pd.read_csv("../input/pubg-finish-placement-prediction/PUBG Finish Place Prediction.csv"))

pub2=reduce_mem_usage(pd.read_csv("../input/pubgplayerstats/PUBG_Player_Statistics.csv"))

pub3=reduce_mem_usage(pd.read_csv("../input/deprecated-pubg-finish-placement-prediction/train.csv"))

pub4=reduce_mem_usage(pd.read_csv("../input/deprecated-pubg-finish-placement-prediction/test.csv"))
train_df.head()
pub1.head()
pub2.head()
pub3.head()
test_df.head()
pub3.columns
pub3=pub3.drop(['groupId','killPlace', 'killPoints', 'kills','DBNOs','matchId','boosts', 'damageDealt'], axis=1)
pub3=pub3.drop("winPlacePerc",axis=1)
pub3 = pub3.rename(columns={'Id': 'soldierId'})

merge_new=pd.merge(train_df,pub3,on="soldierId",how="outer")
merge_new.shape
pub4 = pub4.rename(columns={'Id': 'soldierId'})

pub4=pub4.drop(['groupId','killPlace', 'killPoints', 'kills','DBNOs','matchId','boosts', 'damageDealt'],axis=1)
merge_new_t=pd.merge(test_df,pub3,on="soldierId",how="left")
merge_new.columns
merge_new_t.columns

y=merge_new['bestSoldierPerc']

merge_new=merge_new.drop('bestSoldierPerc',axis=1)
#train_df=train_df.drop(["soldierId"],axis=1)

merge_new_t=merge_new_t.drop(["Unnamed: 0","index"],axis=1)
# from sklearn.preprocessing import StandardScaler



# std=StandardScaler()

# train_df_scaled=std.fit_transform(merge_train)

# test_df_scaled=std.fit_transform(merge_test)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(merge_new,y,random_state=0,test_size=0.20)
print(X_train.shape)

print(X_test.shape)
from sklearn.ensemble import RandomForestRegressor



reg=RandomForestRegressor()

reg.fit(X_train,y_train)

print(reg.score(X_train,y_train))

print(reg.score(X_test,y_test))
merge_new_t.fillna(0,inplace=True)
predictions=reg.predict(merge_new_t)
submission=pd.read_csv("../input/got-battle-of-blackwater/sample_submission.csv")

submission["bestSoldierPerc"]=predictions

submission.to_csv("random.csv",index=False)