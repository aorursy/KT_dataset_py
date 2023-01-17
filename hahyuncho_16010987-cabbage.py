# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train  = pd.read_csv('/kaggle/input/mlregression-cabbage-price/train_cabbage_price.csv')

test = pd.read_csv('/kaggle/input/mlregression-cabbage-price/test_cabbage_price.csv')

submit = pd.read_csv('/kaggle/input/mlregression-cabbage-price/sample_submit.csv')

train1 = train

train.year
train1['year'] = train['year'].astype(str)

train1['day'] = train['year'].str[-2:]

train1['month'] = train['year'].str[4:6]

train1['year'] = train['year'].str[0:4]

train1['year'] = train1['year'].astype(int)



train1['month'] = train1['month'].str.lstrip("0")    

train1['month'] = train1['month'].astype(int)



train1['day'] = train1['day'].str.lstrip("0")   

train1['day'] = train['day'].astype(int)

train1
train1.dtypes

test1 = test
test1['year'] = test['year'].astype(str)

test1['day'] = test['year'].str[-2:]

test1['month'] = test['year'].str[4:6]

test1['year'] = test['year'].str[0:4]

test1['year'] = test1['year'].astype(int)



test1['month'] = test1['month'].str.lstrip("0")    

test1['month'] = test1['month'].astype(int)



test1['day'] = test1['day'].str.lstrip("0")   

test1['day'] = test['day'].astype(int)

test1


X = train1.drop('avgPrice',axis=1)

Y = train['avgPrice']



from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(X)

X_train_std = sc.transform(X)

X_test1_std = sc.transform(test1)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X_train_std,Y, test_size=0.3, random_state=1)

from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import accuracy_score



regressor = KNeighborsRegressor(n_neighbors=30, weights = "distance")

regressor.fit(X_train,y_train)

guess = regressor.predict(X_test1_std)

RMSE = mean_squared_error(guess,submit['Expected'])**0.5

print(RMSE)

from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import accuracy_score



regressor = KNeighborsRegressor(n_neighbors=100, weights = "distance")

regressor.fit(X_train,y_train)

regressor.fit(X_test,y_test)

guess = regressor.predict(X_test1_std)

RMSE = mean_squared_error(guess,submit['Expected'])**0.5

print(RMSE)

submit['Expected'] = guess

submit=submit.astype(np.int32)

submit.to_csv('sample_submit.csv', mode='w', header= True, index= False)
submit