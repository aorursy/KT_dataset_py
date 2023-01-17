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
train = pd.read_csv('../input/train.csv')

test =  pd.read_csv('../input/test.csv')

train.head()

train.columns

train.info()

train.describe()





#проверяем размерность датасета 

print(train.shape)


import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score,r2_score



train_features = ['LotFrontage', 'OverallCond', 'OverallQual','LotArea','YearBuilt','YearRemodAdd','GrLivArea']



X_train = train[train_features]



y_train = train['SalePrice'].copy()







X_train.head()

# X_train = pd.concat([train.loc[:2400, numeric_features], raw.iloc[:2401, 16:]], axis=1)



X_train['LotFrontage'].mean()


# заменим на среднее

X_train.describe()



X_train['LotFrontage'].loc[pd.isnull(X_train['LotFrontage']) == True] = X_train['LotFrontage'].mean()



X_train['LotFrontage']



X_train.head()

# X_train['LotFrontage'][type(X_train['LotFrontage']) != float]
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train,test_size=0.3, random_state = 17)

# print(y_train.shape[0],df.shape[0])

print(X_train.shape, X_valid.shape)
from sklearn import linear_model

regressor = linear_model.LinearRegression(n_jobs=-1) #, random_state = 17)



regressor.fit(X=X_train, y=y_train)



y_predict = regressor.predict(X_valid)



y_predict = pd.Series(y_predict)

y_predict

y_valid = pd.Series(y_valid)

y_valid.reset_index(drop=True, inplace=True)

y_valid
print(y_valid.shape, y_predict.shape)
compare_df = pd.concat([y_valid, y_predict], axis=1)



compare_df
r2_score(y_pred=y_predict, y_true=y_valid)

# # X_train = X_train.drop(['SalePrice'], axis = 'columns')



# X_train.head()
y_train.head()
train.head()

train['LotFrontage'].value_counts().sort_index().plot.line()
train['OverallQual'].value_counts().sort_index().plot.bar()
train['OverallCond'].value_counts().sort_index().plot.bar()