import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline

import seaborn as sns

import random
train = pd.read_csv('../input/train.csv')



test = pd.read_csv('../input/test.csv')
train.info()
train.dtypes
train.head() 
train.describe()
test.info()
test.dtypes
train.isnull().values.any()
test.isnull().values.any()
train.duplicated().sum()
import seaborn as sns



fig, axs = plt.subplots(figsize=(10, 8))



corr = train.corr()



sns.heatmap(corr, center=0)
corr
X=train.drop('AveragePrice',axis=1)

y=train['AveragePrice']

y.describe()
X.head()
testing = test
from sklearn import linear_model



y = np.array(y)



X = np.array(X)



testing = np.array(testing)
testing
X
y
regression = linear_model.LinearRegression()



regression.fit(X, y)
y_pred = regression.predict(testing)
from sklearn.ensemble import RandomForestRegressor





regressor = RandomForestRegressor(n_estimators=1000,max_depth=100,n_jobs=1).fit(X, y)

y_pred = regressor.predict(testing)
ID = X_test[:,0] 





m = np.array(ID)

n = np.array(y_pred)

o = [m,n]





pd.DataFrame(o).transpose().to_csv("kernelup.csv", index = 0)