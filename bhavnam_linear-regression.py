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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
houses = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
houses.head()
houses.describe()
houses['price'].describe()
houses.isnull().sum()
corr = houses.corr()
sns.heatmap(corr)
plt.figure(figsize=(10,10))
sns.distplot(houses['price'])
sns.barplot(x=houses['sqft_living'], y=houses['price'])
#plt.xticks(rotation= 45)
plt.xlabel('Area')
plt.ylabel('Price')
plt.figure(figsize=(10,10))
sns.jointplot(x=houses['sqft_living'], y=houses['price'], data=houses, kind='reg')
sns.jointplot(houses.sqft_living, houses.price, kind="kde")
cols = ['sqft_living']
x = houses[cols]
y = houses.price.values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2) 
lr = LinearRegression()
lr.fit(x_train, y_train)
print (lr.intercept_)
print (lr.coef_)
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, lr.predict(x_test), color = 'blue')
plt.xlabel('Area (in sqft)')
plt.ylabel('Price')
plt.show()
from sklearn.metrics import r2_score

r = r2_score(y_test, lr.predict(x_test))
mse = mean_squared_error(y_test, lr.predict(x_test))

print(r)
print(np.sqrt(mse))
print(cross_val_score(lr, x , y, cv=3))
import statsmodels.api as sm

X = sm.add_constant(x)  # Adds a constant term to the predictor
X.head()

est=sm.OLS(y, X)
est = est.fit()
est.summary()
cols = ['grade','sqft_living','floors','bedrooms']
x=np.array(houses[cols])
y=np.array(houses['price'])
plt.figure(figsize=(10,10))
sns.jointplot(x=houses['grade'], y=houses['price'], data=houses, kind='reg')
plt.figure(figsize=(10,10))
sns.jointplot(x=houses['floors'], y=houses['price'], data=houses, kind='reg')
plt.figure(figsize=(10,10))
sns.jointplot(x=houses['bedrooms'], y=houses['price'], data=houses, kind='reg')
lr=LinearRegression()
lr=lr.fit(x,y)
y_pred=lr.predict(x)

rmse=np.sqrt(mean_squared_error(y,y_pred))
r2=lr.score(x,y)
print(rmse)
print(r2)