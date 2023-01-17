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

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

from sklearn.linear_model import LinearRegression

import scipy, scipy.stats

import statsmodels.formula.api as sm

from sklearn import linear_model

warnings.filterwarnings('ignore')

%matplotlib inline
df = pd.read_csv('../input/train-data/train_data.csv')
df.head()
df.tail()
df.columns
df.shape
df.info()
df_new = df.iloc[:,np.r_[0:10,14:19]]
df_new.head()
corrmat = df_new.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.9);
corr=df_new.corr()["total_No_of_medals"]

corr
sns.distplot(df_new['total_No_of_medals'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_new['total_No_of_medals'], plot=plt)
sns.jointplot(df_new['total_No_of_medals'],df_new['gdp_nominal - US$MM'],color='gold');
from sklearn.linear_model import LinearRegression

import scipy, scipy.stats

import statsmodels.formula.api as sm
null_counts = df_new.isnull().sum()

null_counts[null_counts > 0].sort_values
df_new['gdp_nominal - US$MM'].fillna(df_new['gdp_nominal - US$MM'].median(), inplace=True)
df_new['gdp_per capita (usd)'].fillna(df_new['gdp_per capita (usd)'].median(), inplace=True)
sns.distplot(df_new['gdp_per capita (usd)'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_new['gdp_per capita (usd)'], plot=plt)
sns.distplot(df_new['population'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_new['population'], plot=plt)
sns.distplot(df_new['no_of_internet_users'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_new['no_of_internet_users'], plot=plt)
plt.scatter(df_new["no_of_internet_users"],df_new.total_No_of_medals, color='red')

plt.title("Scatter plot between internet users and number of models")
plt.scatter(df_new["population"],df_new.total_No_of_medals, color='red')

plt.title("Scatter plot between population and number of models")
X = df_new[['no_of_internet_users','population','gdp_nominal - US$MM','gdp_per capita (usd)']]
Y = df_new.iloc[:,-1]
from sklearn import linear_model

from sklearn.linear_model import LinearRegression

import scipy, scipy.stats

import statsmodels.api as sm
df_new.shape
result = sm.OLS(Y,X).fit()

result.summary()
x = df_new[['no_of_internet_users','population','gdp_nominal - US$MM','host_n','Gender gap index']].values

y = df_new.iloc[:,-1].values
corr = np.corrcoef(x,rowvar=0)

corr
w,v = np.linalg.eig(corr)

w
v[:,3]

v[:,4]
df_new.columns

df_1 = df_new.iloc[:,np.r_[2:7,8:10]].values
X1 = np.append(arr = np.ones((105,1)).astype(int), values = df_1, axis = 1)
x_1 = X1[:,[0,1,2,3,4,5,6,7]]
reg_1 = sm.OLS(y,x_1).fit()

reg_1.summary()
#second model

x_2 = X1[:,[1,2,3,4,5,6,7]]

reg_2 = sm.OLS(y,x_2).fit()

reg_2.summary()
#third model

x_3 = X1[:,[1,2,3,5,6,7]]

reg_3 = sm.OLS(y,x_3).fit()

reg_3.summary()
x_5 = X1[:,[0,1,3,5,6,7]]

reg_5 = sm.OLS(y,x_5).fit()

reg_5.summary()
from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

import scipy, scipy.stats

import statsmodels.formula.api as sm
x = df_new[['gdp_nominal - US$MM','population','host_n','no_of_internet_users','Gender gap index']].values

y = df_new.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)
X_test.shape
import statsmodels.api as sm

result = sm.OLS(y_train,X_train).fit()

result.summary()
y_pred = result.predict(X_test)
y_pred
y_test
plt.scatter(result.predict(X_test),y_test)

plt.show()
np.mean((result.predict(X_test) - y_test))
np.mean((result.predict(X_test) - y_test)**2)
fig = plt.figure()

res = stats.probplot(y_test - y_pred, plot=plt)
plt.plot(result.predict(X_test), y_test - result.predict(X_test), 'bo')

plt.axhline(y=0, color='black')

plt.title('Linear')

plt.xlabel('predicted values')

plt.ylabel('residuals')

plt.show()
##cook's distance

influence = result.get_influence()



(cook, p) = influence.cooks_distance

plt.stem(np.arange(len(cook)), cook, markerfmt=",")
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from math import sqrt

rmse_ols_withoutscaling= sqrt(mean_squared_error(y_test, y_pred))

rmse_ols_withoutscaling
mean_squared_error(y_test, y_pred)
mean_absolute_error(y_test, y_pred)
y = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

y
# fit a model

lm = linear_model.LinearRegression()



model1 = lm.fit(X_train, y_train)

predictions = lm.predict(X_test)
model1.score(X_test, y_test)
plt.scatter(y_test, predictions)

plt.xlabel("Actual Values")

plt.ylabel("Predictions")
df_new[df_new.dtypes[(df.dtypes=="float64")|(df.dtypes=="int64")].index.values].hist(figsize=[12,12])
#Importing MinMaxScaler and initializing it

from sklearn.preprocessing import MinMaxScaler

minmax=MinMaxScaler()

#Scaling down both train and test data set

X_trainminmax=minmax.fit_transform(X_train)

X_testminmax=minmax.fit_transform(X_test)
result1 = sm.OLS(y_train,X_trainminmax).fit()

result1.summary()
y_pred1 = result1.predict(X_testminmax)
y_pred1
y_test
from math import sqrt

rmse_ols = sqrt(mean_squared_error(y_test, y_pred1))

rmse_ols
##line plot

sns.kdeplot(y_pred1, label = 'Predictions')

sns.kdeplot(y_test,label = 'actual_Values')
##gradient boosting regressor

from sklearn.ensemble import GradientBoostingRegressor



c = GradientBoostingRegressor()



c.fit(X_train,y_train)
pred = c.predict(X_test)

mae = np.mean(abs(pred - y_test))
mae
from math import sqrt

rmse_GB = sqrt(mean_squared_error(y_test, pred))

rmse_GB
pred
y_test
plt.scatter(y_test, pred)

plt.xlabel("True Values")

plt.ylabel("Predictions")
sns.kdeplot(pred, label = 'Predictions')

sns.kdeplot(y_test,label = 'actual_Values')
from sklearn.ensemble import RandomForestRegressor

c = RandomForestRegressor(random_state = 30)

c.fit(X_train,y_train)

pred = c.predict(X_test)

c
mean_squared_error(y_test, pred)
mae = np.mean(abs(pred - y_test))

mae
from math import sqrt

rmse_RF = sqrt(mean_squared_error(y_test, pred))

rmse_RF
##line plot

sns.kdeplot(pred, label = 'Predictions')

sns.kdeplot(y_test,label = 'actual_values')
from sklearn.neighbors import KNeighborsRegressor

c = KNeighborsRegressor(n_neighbors = 2)
c.fit(X_train,y_train)

c
pred = c.predict(X_test)

mae = np.mean(abs(pred - y_test))

mean_squared_error(y_test, pred)
mae = np.mean(abs(pred - y_test))

mae
from math import sqrt

rmse_KNN = sqrt(mean_squared_error(y_test, pred))

rmse_KNN
plt.scatter(y_test, pred)

plt.xlabel("True Values")

plt.ylabel("Predictions")
residual = y_test - pred

plt.scatter(pred,residual)
figsize = (8,8)



r = pred - y_test

plt.hist(r,color = 'red', bins = 20,edgecolor = 'black')

plt.xlabel('Residuals')

plt.title('Dist of Residuals')
##line plot

sns.kdeplot(pred, label = 'Predictions')

sns.kdeplot(y_test,label = 'actual_values')
###k-fold cross validation

from sklearn.model_selection import cross_val_score

efficiency = cross_val_score(estimator = c, X= X_train, y = y_train, cv = 10)

efficiency
efficiency.mean()
from sklearn.model_selection import GridSearchCV

parameters = [{'algorithm': ['auto','ball_tree']# ‘kd_tree’, ‘brute’],

               ,'n_neighbors':[2,3,4,5],

               'leaf_size':[20,25,20,35],'p':[1,2],'n_jobs':[-1]}]

grid_s = GridSearchCV(estimator = c,param_grid = parameters,scoring = 'r2',cv = 10,n_jobs = -1)

grid_s = grid_s.fit(X_train,y_train)

best_param = grid_s.best_params_
best_param
best_accuracy = grid_s.best_score_

best_accuracy
G = pd.DataFrame({'RMSE': [rmse_KNN,rmse_RF,rmse_GB,rmse_ols,rmse_ols_withoutscaling],'model_name':['K-nearest Neighbors','Random Forest','Gradient Boosting','Ordinary LS','Ordinary LS without scaling']})

ax = plt.subplot()

ax.set_title('Comparison of Models Accuracy')

G.groupby('model_name').mean()['RMSE'].plot(kind='bar',figsize=(10,8), ax = ax,color = ('green','red','green','green','green'))
df_test = pd.read_csv('../input/testdata/test_data_2020_olympics.csv')
df_test.info()
o_2020 = df_test.iloc[:,np.r_[2:3,4:5,6:7,8:10]]

null_counts = o_2020.isnull().sum()

null_counts[null_counts > 0].sort_values
#Treating Missing Values

df_test['gdp_nominal - US$MM'].fillna(df_test['gdp_nominal - US$MM'].median(), inplace=True)
##prediction of gold medals on validation sample

from sklearn.neighbors import KNeighborsRegressor

c = KNeighborsRegressor(n_neighbors = 2)
total_p =  df_test.iloc[:,np.r_[2:3,4:5,6:7,8:10]].values
total_p
c.fit(X_train,y_train)

pred_t = c.predict(total_p)
pred_t
medals_table = pd.DataFrame(pred_t,columns=['Total Medals'])
medals_table
from sklearn.neighbors import KNeighborsRegressor

c = KNeighborsRegressor(n_neighbors = 2)
y = df_new.iloc[:,11].values
y
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)
gold_model = c.fit(X_train,y_train)

pred_gold = c.predict(X_test)

mae = np.mean(abs(pred_gold - y_test))

mean_squared_error(y_test, pred_gold)
mae = np.mean(abs(pred_gold - y_test))

mae
from math import sqrt

rmse_KNN = sqrt(mean_squared_error(y_test, pred_gold))

rmse_KNN
pred_gold
y_test
pred_gold = c.predict(total_p)
pred_gold
y = df_new.iloc[:,12].values
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)
c.fit(X_train,y_train)

pred_silver = c.predict(X_test)

mae = np.mean(abs(pred_silver - y_test))

mean_squared_error(y_test, pred_silver)
mae = np.mean(abs(pred_silver - y_test))

mae
from math import sqrt

rmse_KNN = sqrt(mean_squared_error(y_test, pred_silver))

rmse_KNN
pred_silver
y_test
pred_silver = c.predict(total_p)
pred_silver
y = df_new.iloc[:,13].values
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)
c.fit(X_train,y_train)

pred_bronze = c.predict(X_test)

mae = np.mean(abs(pred_bronze - y_test))

mean_squared_error(y_test, pred_bronze)
mae = np.mean(abs(pred_bronze - y_test))

mae
from math import sqrt

rmse_KNN = sqrt(mean_squared_error(y_test, pred_bronze))

rmse_KNN
pred_bronze
y_test
pred_bronze = c.predict(total_p)
pred_bronze
medals_table['gold medals'] =pred_gold

medals_table['silver medals'] =pred_silver

medals_table['bronze medals'] =pred_bronze

medals_table['country']=df_new['country']
medals_table