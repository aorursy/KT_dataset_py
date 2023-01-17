# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#!pip install pubg-python



import matplotlib  as plt

import seaborn as sns

from matplotlib import pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv',nrows = 3000000)

# Thanks to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage



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

#        else:

#            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
df_train = reduce_mem_usage(df)

df.shape
df.dropna(inplace=True)
#f, ax = plt.subplots(figsize=(10, 8))

#corr = df.corr()

#sns.heatmap(corr,

#           xticklabels=corr.columns.values,

#           yticklabels=corr.columns.values)

x = df.drop(['Id', 'groupId', 'matchId','matchType','winPlacePerc'],axis=1)



y = df.winPlacePerc

del df
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score

random_forest = RandomForestRegressor(random_state=42, n_jobs=-1,n_estimators = 50)

random_forest.fit(x,y)
#rf = RandomForestRegressor()

#rf.fit(x,y)

#names = x.columns



#sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 

#             reverse=True)
test =  pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')

test.head(5)
x_test = test.drop(['Id', 'groupId', 'matchId','matchType'],axis=1)

y_pred = random_forest.predict(x_test)  # test the output by changing values 

submission = pd.DataFrame({"Id":test['Id'], "winPlacePerc":y_pred})

submission.to_csv("submission_new.csv", index=False)
print('CODE ENEDED')