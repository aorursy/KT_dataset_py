import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

%matplotlib inline
train = pd.read_csv('../input/train.csv')
train.head()

test = pd.read_csv('../input/test.csv')
train.describe()
test
test.info()
train.info()
train.dtypes
train.isnull().values.any()
test.isnull().values.any()
train.dtypes



test
train
train.duplicated().sum()
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = train.corr()

sns.heatmap(corr, center=0)
train1 = train.drop("ID", axis=1)



train1=train1.drop('Total Bags', axis=1)



test1=test.drop('Total Bags', axis=1)

test1=test1.drop('ID', axis=1)

X=train1.drop('AveragePrice',axis=1)

y=train1['AveragePrice']

X.head()
y.describe()
testing = test1
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn import linear_model



y_train = np.array(y_train)



X_train = np.array(X_train)



testing = np.array(testing)
regression = linear_model.LinearRegression()



regression.fit(X_train, y_train)
y_pred = regression.predict(testing)
#from sklearn.preprocessing import StandardScaler



#sc = StandardScaler()  

#X_train = sc.fit_transform(X_train)  

#X_test = sc.transform(X_test)  
from sklearn.ensemble import RandomForestRegressor





regressor = RandomForestRegressor(n_estimators=1000,max_depth=100,n_jobs=1).fit(X_train, y_train)

y_pred = regressor.predict(X_test)
from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
regressor.score(X_test,y_test)


y_pred=regressor.predict(testing)
#ID = test[:,0]

m=test['ID']
a = np.array(m)

b = np.array(y_pred)

p = [a,b]

pd.DataFrame(p).transpose().to_csv("HELLO.csv", index = 0)