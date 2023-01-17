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
train=pd.read_csv("/kaggle/input/replace-reg-31/replace_reg/training_replace_features.csv")

test=pd.read_csv("/kaggle/input/replace-reg-31/replace_reg/validation_replace_features.csv")

print(train.shape)
train.head()
test.head()
train.drop(['ID'],axis=1,inplace=True)

test.drop(['ID'],axis=1,inplace=True)
train.describe()
import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats
from datetime import datetime

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

quantitative.remove('Price')
test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01

normal = pd.DataFrame(train[quantitative])

normal = normal.apply(test_normality)

print(not normal.any())
plt.subplots(figsize=(12,9))

sns.distplot(train['Price'], fit=stats.norm)



# Get the fitted parameters used by the function



(mu, sigma) = stats.norm.fit(train['Price'])



# plot with the distribution



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

stats.probplot(train['Price'], plot=plt)

plt.show()
train.head()
train_corr = train.select_dtypes(include=[np.number])

train_corr.shape
corr = train_corr.corr()

plt.subplots(figsize=(30,30))

sns.heatmap(corr, annot=True)
train["Price"] = np.log(train["Price"])

test["Price"] = np.log(test["Price"])
x_train =train.drop('Price', axis = 1)

y_train=train['Price']
plt.subplots(figsize=(12,9))

sns.distplot(test['Price'], fit=stats.norm)



# Get the fitted parameters used by the function



(mu, sigma) = stats.norm.fit(test['Price'])



# plot with the distribution



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

stats.probplot(test['Price'], plot=plt)

plt.show()
plt.subplots(figsize=(12,9))

sns.distplot(train['Price'], fit=stats.norm)



# Get the fitted parameters used by the function



(mu, sigma) = stats.norm.fit(train['Price'])



# plot with the distribution



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')



#Probablity plot



fig = plt.figure()

stats.probplot(train['Price'], plot=plt)

plt.show()
x_test =test.drop('Price', axis = 1)

y_test=test['Price']
from sklearn.model_selection import GridSearchCV
model = GradientBoostingRegressor()

parameters = {'learning_rate': [0.01, 0.02, 0.03],

                  'subsample'    : [0.9, 0.5, 0.2],

                  'n_estimators' : [50,100, 500, 1000],

                  'max_depth'    : [4, 6,8] 

                 }

grid = GridSearchCV(estimator=model, param_grid = parameters, cv =10, n_jobs=-1)

grid.fit(x_train, y_train)   
y_pred=grid.predict(x_test)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
def mean_absolute_percentage_error(y_test, y_pred1): 

    y_true, y_pred = np.array(y_test), np.array(y_pred1)

    return np.mean(np.abs((y_test - y_pred1) / y_test)) * 100
mean_absolute_percentage_error(y_test, y_pred)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

model1 = RandomForestRegressor()

param_grid={

            'max_depth': [4, 6, 8],

            'n_estimators': (50, 100, 500, 1000),

            'max_features': (2,4,6)

        }

grid_RF = GridSearchCV(estimator=model1, param_grid =param_grid, cv =10, n_jobs=-1)

grid_RF.fit(x_train, y_train)   
y_pred1=grid_RF.predict(x_test)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))
def mean_absolute_percentage_error(y_test, y_pred1): 

    y_true, y_pred = np.array(y_test), np.array(y_pred1)

    return np.mean(np.abs((y_test - y_pred1) / y_test)) * 100
mean_absolute_percentage_error(y_test, y_pred1)
from sklearn.linear_model import LinearRegression

lr =LinearRegression()

parameters_l = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}



lr_gs = GridSearchCV(lr,parameters_l, cv=10)

lr_gs.fit(x_train, y_train)
y_pred2=lr_gs.predict(x_test)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred2))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred2))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
mean_absolute_percentage_error(y_test, y_pred2)
XG=XGBRegressor()

xg_param_grid = {

              

              'learning_rate': [0.01, 0.02, 0.03],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8]

              }

grid_XG = GridSearchCV(estimator=XG, param_grid =xg_param_grid, cv =10, n_jobs=-1)

grid_XG.fit(x_train, y_train)   
y_pred3=grid_XG.predict(x_test)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred3))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred3))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))
mean_absolute_percentage_error(y_test, y_pred3)
LGB=LGBMRegressor()

gridParams_lgb = {

    'learning_rate': [0.01, 0.02, 0.03],

   

    'n_estimators': [40],

    'num_leaves': [6,8,12,16],

    

    'objective' : ['regression'],

    'random_state' : [0], # Updated from 'seed'

    'colsample_bytree' : [0.65, 0.66],

    'subsample' : [0.7,0.75],

    'reg_alpha' : [1,1.2],

    'reg_lambda' : [1,1.2,1.4],

    }

grid_LGB = GridSearchCV(estimator=LGB, param_grid =gridParams_lgb,cv =10, n_jobs=-1)

grid_LGB.fit(x_train, y_train)   
y_pred4=grid_LGB.predict(x_test)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred4))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred4))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred4)))
mean_absolute_percentage_error(y_test, y_pred4)
stack_gen = StackingCVRegressor(regressors=(grid_LGB,grid_XG,lr_gs, grid_RF),

                                meta_regressor=grid_RF,

                                use_features_in_secondary=True)
stack_gen.fit(x_train, y_train) 
y_pred5=stack_gen.predict(x_test)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred5))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred5))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Accuracy --> ",stack_gen.score(x_test, y_test)*100)