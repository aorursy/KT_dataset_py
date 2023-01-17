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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

data = pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv", parse_dates=['date'])
data.tail()
data.info()
data.describe()
data.columns
data['years_renovated'] = data['yr_renovated'] - data['yr_built']
numeric_data = ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']
numeric_analysis = pd.DataFrame(data[numeric_data]).corr()
sns.heatmap(numeric_analysis)
y_data = data['price']
x_data = data.drop('price', axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=1)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train[['sqft_living']],y_train)

print("R-square is: " , lr.score(x_test[['sqft_living']], y_test))
from sklearn.metrics import mean_squared_error
y_hat = lr.predict(x_test[['sqft_living']])
mse = mean_squared_error(y_test, y_hat)

print("The mean squared error is: ", mse)
plt.figure(figsize=(12,10))
sns.regplot(x='sqft_living',y='price',data=data)
plt.figure(figsize=(12,10))
sns.residplot(data['sqft_living'], data['price'])
newy_train = np.log(y_train)
newy_test = np.log(y_test)
lmr = LinearRegression()
lmr.fit(np.log(x_train[["sqft_living"]]), newy_train)
newy_hat = lmr.predict(x_test[['sqft_living']])
lmr.score(np.log(x_test[['sqft_living']]), newy_test)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

added_features = ['sqft_living','grade', 'sqft_above', 'sqft_living15','bathrooms','view','sqft_basement','lat','waterfront','yr_built','bedrooms','years_renovated']
X_data = data[added_features]
X_data = scaler.fit_transform(X_data)
Y_data = data['price']
X_train,X_test,Y_train,Y_test = train_test_split(X_data,Y_data,test_size=0.20, random_state=1)
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
       
lr_final = LinearRegression()
lr_final.fit(X_train, Y_train)
lr_final.score(X_test,Y_test)
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0))
reg.fit(X_train, Y_train)
reg.score(X_test, Y_test)
from sklearn import linear_model
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(X_train, Y_train)
lasso.score(X_test, Y_test)
from sklearn.linear_model import SGDRegressor
clf = SGDRegressor(eta0=0.1, penalty="l2", max_iter=100)
clf.fit(X_train, Y_train)
clf.score(X_test, Y_test)
from xgboost import XGBRegressor

my_model = XGBRegressor(subsample=0.2 ,gamma=1000,reg_alpha=0.8,reg_lamda=0.8, n_estimators=1000, learning_rate=0.06)
my_model.fit(X_train, Y_train, early_stopping_rounds=5, 
             eval_set=[(X_test, Y_test)], verbose=False)
predictions = my_model.predict(X_test)
print("R2 : " + str(my_model.score(X_test, Y_test)))
from sklearn.metrics import explained_variance_score

explained_variance_score(Y_test, predictions)