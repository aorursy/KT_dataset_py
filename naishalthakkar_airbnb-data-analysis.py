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
import pandas as pd

import numpy as np

import scipy.stats as stats

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn.metrics as sklm

from matplotlib import pyplot as plt

import matplotlib.mlab as mlab

import numpy as np 

import statsmodels.api as sm 

import pylab as py 

import scipy.stats as st

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import RobustScaler

import sklearn.model_selection as GridSearchCV

from sklearn.linear_model import Ridge

import sklearn.model_selection as ms

import math

import xgboost
import pandas as pd

df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

df.head()
df.info()

df.describe()
df['neighbourhood_group'].value_counts()


df['room_type'].value_counts()
df.isnull().sum()
df.drop(['id','host_name','last_review'],axis=1,inplace=True)

df.fillna({'reviews_per_month':0},inplace=True)
df.isnull().sum()
df.info()
n, bins, patches = plt.hist(df.price, 50, normed=1)

mu = np.mean(df.price)

sigma = np.std(df.price)

plt.plot(bins, mlab.normpdf(bins, mu, sigma))
sm.qqplot(df.price,stats.norm,fit=True ,line ='45') 

py.show()
df = df[np.log1p(df['price']) < 8]

df = df[np.log1p(df['price']) > 3]
df.price=np.log1p(df['price'])
df.info()
sm.qqplot(df.price,stats.norm,fit=True ,line ='45') 

py.show()
fig, axes = plt.subplots(1,3, figsize=(21,6))

sns.distplot(df['price'], ax=axes[0])

sns.distplot(np.log1p(df['price']), ax=axes[1])

axes[1].set_xlabel('log(1+price)')

sm.qqplot(np.log1p(df['price']), stats.norm, fit=True, line='45', ax=axes[2]);
st.shapiro(df.price)
pvt = df[df['room_type'] == 'Private room']

share = df[df['room_type'] == 'Shared room']

apt = df[df['room_type'] == 'Entire home/apt']
st.kruskal(pvt.price,share.price,apt.price)
st.f_oneway(pvt.price,share.price,apt.price)
ind = ['Private Rooms','Apartments','Shared Rooms']

x = pd.DataFrame([pvt.price.mean(),apt.price.mean(),share.price.mean()], index=ind)

x
st.f_oneway(pvt.price,share.price,apt.price)
a = df[df['neighbourhood_group'] == 'Brooklyn']

b = df[df['neighbourhood_group'] == 'Manhattan']

c = df[df['neighbourhood_group'] == 'Queens']

d = df[df['neighbourhood_group'] == 'Staten Island']

e = df[df['neighbourhood_group'] == 'Bronx']



st.kruskal(a.price,b.price,c.price,d.price,e.price)
st.f_oneway(a.price,b.price,c.price,d.price,e.price)
st.f_oneway(pvt.minimum_nights,share.minimum_nights,apt.minimum_nights)
ind = ['Private Rooms','Apartments','Shared Rooms']

x = pd.DataFrame([pvt.minimum_nights.mean(),apt.minimum_nights.mean(),share.minimum_nights.mean()], index=ind)

x
st.f_oneway(a.minimum_nights,b.minimum_nights,c.minimum_nights,d.minimum_nights,e.minimum_nights)
ind = ['Brooklyn','Manhattan','Queens','Staten Island','Bronx']

x = pd.DataFrame([a.minimum_nights.mean(),b.minimum_nights.mean(),c.minimum_nights.mean(),d.minimum_nights.mean(),e.minimum_nights.mean()], index=ind)

x
X=df[['neighbourhood_group','latitude','longitude','room_type','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]

X=pd.get_dummies(X, columns=['neighbourhood_group','room_type'])
y=df['price']
X.head()
lm=LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler= RobustScaler()

# transform "x_train"

X_train = scaler.fit_transform(X_train)

# transform "x_test"

X_test = scaler.transform(X_test)



lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
# Evaluated metrics



mae = metrics.mean_absolute_error(y_test, predictions)

mse = metrics.mean_squared_error(y_test, predictions)

rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

r2 = metrics.r2_score(y_test, predictions)



print('MAE (Mean Absolute Error): %s' %mae)

print('MSE (Mean Squared Error): %s' %mse)

print('RMSE (Root mean squared error): %s' %rmse)

print('R2 score: %s' %r2)
# Avtual vs predicted values



error = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'Predicted Values': predictions.flatten()})

error.head()
ridge=Ridge()

parameters= {'alpha':[x for x in range(1,101)]}



ridge_reg=ms.GridSearchCV(ridge, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)

ridge_reg.fit(X_train,y_train)

print("The best value of Alpha is: ",ridge_reg.best_params_)

print("The best score achieved with Alpha=14 is: ",math.sqrt(-ridge_reg.best_score_))
ridge_mod=Ridge(alpha=19)

ridge_mod.fit(X_train,y_train)

y_pred_train=ridge_mod.predict(X_train)

y_pred_test=ridge_mod.predict(X_test)



print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(y_train, y_pred_train))))

print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, y_pred_test))))  
from sklearn.linear_model import Lasso



parameters= {'alpha':[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}



lasso=Lasso()

lasso_reg=ms.GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=15)

lasso_reg.fit(X_train,y_train)



print('The best value of Alpha is: ',lasso_reg.best_params_)
lasso_mod=Lasso(alpha=0.0009)

lasso_mod.fit(X_train,y_train)

y_lasso_train=lasso_mod.predict(X_train)

y_lasso_test=lasso_mod.predict(X_test)



print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(y_train, y_lasso_train))))

print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, y_lasso_test))))

r2 = metrics.r2_score(y_test, y_lasso_test)

print('R2 score: %s' %r2)
coefs = pd.Series(lasso_mod.coef_, index = X.columns)



imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh", color='yellowgreen')

plt.xlabel("Lasso coefficient", weight='bold')

plt.title("Feature importance in the Lasso Model", weight='bold')

plt.show()
from xgboost.sklearn import XGBRegressor
xgb= XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.5, gamma=0,

             importance_type='gain', learning_rate=0.01, max_delta_step=0,

             max_depth=3, min_child_weight=0, missing=None, n_estimators=4000,

             n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,

             reg_alpha=0.0001, reg_lambda=0.01, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)

xgmod=xgb.fit(X_train,y_train)

xg_pred=xgmod.predict(X_test)

print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(y_test, xg_pred))))

r2 = metrics.r2_score(y_test, xg_pred)

print('R2 score: %s' %r2)
from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor(n_estimators=1000, random_state=0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

r2 = metrics.r2_score(y_test, y_pred)

print('R2 score: %s' %r2)
feat_importances = pd.Series(regressor.feature_importances_, index=X.columns)

feat_importances.plot(kind='barh')