import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

import folium

from folium.plugins import HeatMap



# Import data

data = pd.read_csv('../input/housingBoston.csv', encoding='latin-1')



# Peek

data.head()
data.tail()
data.isnull().sum()
data.dtypes
data.describe()
data.shape
data['medv'].hist()
data['LogMedv'] = np.log(data['medv']) #Log transformation of output

plt.hist(data['LogMedv']) 
data.head()
data.corr()
x = data.iloc[:,0:13] #features

y = data['LogMedv'] #labels
z = x.drop(['ptratio','nox','tax','rm','age'], axis = 1) ##feature selection
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(z.values, i) for i in range(z.shape[1])]

vif['features'] = z.columns
vif
z.corr()
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.20, random_state = 5)
xTrain.shape #80%
xTest.shape #20%
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(xTrain, yTrain)
model.intercept_
model.coef_
z.columns
model.score(xTest, yTest) #R squared coefficient determination
xTest
pred = model.predict(xTest)
pred
yTest
from sklearn import metrics

print('MAE', metrics.mean_absolute_error(yTest, pred))

print('MSE', metrics.mean_squared_error(yTest, pred))

print('RMSE', np.sqrt(metrics.mean_squared_error(yTest, pred)))