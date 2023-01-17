# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("./"))



# Any results you write to the current directory are saved as output.
PATH = "../input/"

train_df = pd.read_csv(f'{PATH}train.csv', index_col = 'Id')

test_df = pd.read_csv(f'{PATH}test.csv', index_col ='Id')

train_df.head()
from sklearn.ensemble import RandomForestRegressor
target = train_df['SalePrice']

train_df = train_df.drop('SalePrice', axis =1)





train_df['training_set'] =  True

test_df['training_set'] =  False



df_full = pd. concat([train_df, test_df])

df_full = df_full.interpolate()

df_full = pd.get_dummies(df_full)



train_df = df_full[df_full['training_set']==True]

test_df = df_full[df_full['training_set']==False]



train_df = train_df.drop('training_set', axis =1)

test_df= test_df.drop('training_set', axis =1)
rf = RandomForestRegressor(n_estimators = 200, n_jobs = -1)

rf.fit(train_df, target)
preds = rf.predict(test_df)
my_submission = pd.DataFrame({'Id':test_df.index, 'SalePrice': preds})

my_submission.to_csv('./submission.csv', index = False)