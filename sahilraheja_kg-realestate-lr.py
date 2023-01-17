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
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
sns.set()
data = pd.read_csv("../input/real-estate-price-prediction/Real estate.csv")
data.head()
data.shape
data.describe()
data.isna().sum()
data.dtypes
data.corr()
sns.regplot(x="X1 transaction date", y="Y house price of unit area", data=data)
sns.regplot(x="X2 house age", y="Y house price of unit area", data=data)
sns.regplot(x="X3 distance to the nearest MRT station", y="Y house price of unit area", data=data)
sns.regplot(x="X4 number of convenience stores", y="Y house price of unit area", data=data)
sns.regplot(x="X5 latitude", y="Y house price of unit area", data=data)
sns.regplot(x="X6 longitude", y="Y house price of unit area", data=data)
data['Year'] = data['X1 transaction date'].astype(str).apply(lambda x: x[:4])
data['X3 distance to the nearest MRT station'].describe()
sns.distplot(data['X3 distance to the nearest MRT station'])
#Distance from MRT Station 

Q1 = data['X3 distance to the nearest MRT station'].quantile(0.99)
data1 = data[data['X3 distance to the nearest MRT station']<Q1]
Q1
data1['X3 distance to the nearest MRT station'].describe()
data1['X3 distance to the nearest MRT station'].value_counts()
sns.distplot(data1['Y house price of unit area'])
data.columns
sns.distplot(data['X5 latitude'])
data1.shape
data1['Year'] = data1['X1 transaction date'].astype(str).apply(lambda x: x[:4])
data1
data1.reset_index(drop = True)
data.columns
sns.distplot(data['X1 transaction date'])
sns.distplot(data['X2 house age'])
sns.distplot(data['X3 distance to the nearest MRT station'])
sns.distplot(data['X5 latitude'])
sns.distplot(data['X4 number of convenience stores'])
sns.distplot(data['X5 latitude'])
sns.distplot(data['X6 longitude'])
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, figsize = (15,3))

ax1.scatter(data1['X3 distance to the nearest MRT station'], data1['Y house price of unit area'])
ax1.set_title('Distance and Price')
ax2.scatter(data1['X4 number of convenience stores'], data1['Y house price of unit area'])
ax2.set_title('No of stores and Price')
ax3.scatter(data1['X6 longitude'], data1['Y house price of unit area'])
ax3.set_title('Longitude and Price')

plt.show()
log_dist = np.log(data1['X3 distance to the nearest MRT station'])
data1['Distance_logged'] = log_dist
data1
sns.distplot(data1['Distance_logged'])
X = data1.drop(['No', 'X1 transaction date', 'Y house price of unit area',
               'X5 latitude', #'X4 number of convenience stores',
               'X3 distance to the nearest MRT station',
                'X6 longitude',
              ], axis = 1)
Y = data1['Y house price of unit area']
#Dummify Year column 

titles_dummies = pd.get_dummies(X['Year'], prefix='YR  ')
X = pd.concat([X, titles_dummies], axis=1)
X.drop('Year', axis = 1, inplace = True)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
y_train.shape, x_train.shape
x_train.dtypes
#On train model 

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
#lr.score(x_train, y_train)
lr.score(x_test, y_test)
#y_predict = lm.predict(Z)
Y_predict = lr.predict(X)
data2 = {'Actual Prices': data1['Y house price of unit area'], 
        'Predicted Prices ': Y_predict} 
pred_check = pd.DataFrame(data2)
pred_check
ax2 = sns.distplot(data1['Y house price of unit area'], hist = False, color = 'r', label = 'Actual Value')
sns.distplot(Y_predict, hist = False, color = 'b', label = 'Fitted Values', ax = ax2)
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(data1['Y house price of unit area'], Y_predict))
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

model1=sm.OLS(y_train, x_train)

result=model1.fit()

result.summary()
