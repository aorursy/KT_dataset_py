# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    #start_mem = df.memory_usage().sum() / 1024**2

    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



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



    #end_mem = df.memory_usage().sum() / 1024**2

    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
train_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

test_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
train_df = reduce_mem_usage(train_df)

test_df = reduce_mem_usage(test_df)
from matplotlib import pyplot as plt

import seaborn as sns

import missingno as msno
train_df.drop('Id', axis=1, inplace=True)
cat_features = ['groupId', 'matchId', 'matchType']
num_features = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 'longestKill', 'matchDuration', 'maxPlace', 'numGroups', 'rankPoints', 'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints', 'winPlacePerc']
train_df.nunique()
train_df.describe()
train_df.shape
%%time

msno.bar(train_df, figsize=(20, 10), sort=None)
%%time

train_df[train_df.duplicated()]
EDA_df = train_df.sample(10000, random_state=42)
EDA_df.describe()
for column in num_features:

    plt.figure()

    sns.distplot(EDA_df[column])
sns.pairplot(EDA_df)
for i, feat_a in enumerate(num_features[:-1]):

    train_df[f'log(1 + {feat_a})'] = np.log1p(train_df[feat_a])

    for feat_b in num_features[i:-1]:

        train_df[f'{feat_a}*{feat_b}'] = train_df[feat_a] * train_df[feat_b]
for i, feat_a in enumerate(num_features[:-1]):

    test_df[f'log(1 + {feat_a})'] = np.log1p(test_df[feat_a])

    for feat_b in num_features[i:-1]:

        test_df[f'{feat_a}*{feat_b}'] = test_df[feat_a] * test_df[feat_b]
from lightgbm import LGBMRegressor
model = LGBMRegressor(m_estimators=500)
model.fit(train_df[num_features[:-1]], train_df[num_features[-1]])
test_df[num_features[-1]] = model.predict(test_df[num_features[:-1]])
test_df[['Id', num_features[-1]]].to_csv('submission.csv', index=False)