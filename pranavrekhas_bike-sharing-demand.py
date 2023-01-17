import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

%matplotlib inline 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV , LassoCV, Ridge , Lasso

import scipy.stats as st
df = pd.read_csv('../input/bike-sharing-demand/train.csv')

df1 = pd.read_csv('../input/bike-sharing-demand/test.csv')

sample_submission_df = pd.read_csv('../input/bike-sharing-demand/sampleSubmission.csv')

print(df.shape)

df
df1.head()
df.dtypes
df['datetime'].value_counts()
df['dayofweek'] = pd.DatetimeIndex(df['datetime']).dayofweek

df.tail(2)
df['month'] = pd.DatetimeIndex(df['datetime']).month

df.tail(2)
df = df.drop(['datetime','casual','registered'],axis = 1)
sns.distplot((df['count']**2))

plt.show()
sns.distplot(df['count'])

plt.show()
(np.log(df['count']**2)).isnull().sum()
#sns.pairplot(df)
df.corr()
df['season'].value_counts()
c = df.groupby(by='season')

c1 = c.get_group(4)

c2 = c.get_group(3)

c3 = c.get_group(2)

c4 = c.get_group(1)
from scipy.stats import f_oneway, ttest_ind

f_oneway(c1['count'],c2['count'],c3['count'],c4['count'])
df.boxplot(column = 'count', by = 'holiday')

plt.show()
h = df.groupby(by='holiday')

h1 = h.get_group(1)

h0 = h.get_group(0)
ttest_ind(h1['count'], h0['count'])
df.boxplot(column = 'count', by = 'workingday')
w = df.groupby(by='workingday')

w1 = w.get_group(1)

w0 = w.get_group(0)
ttest_ind(w1['count'], w0['count'])
df.boxplot(column = 'count', by = 'weather')

plt.show()
wh = df.groupby(by='weather')

wh1 = wh.get_group(4)

wh2 = wh.get_group(3)

wh3 = wh.get_group(2)

wh4 = wh.get_group(1)
f_oneway(wh1['count'],wh2['count'],wh3['count'],wh4['count'])
df.columns
df = df.drop(['holiday','workingday', 'month'], axis = 1)
df.columns
#sns.pairplot(df)
df.corr()
from sklearn.preprocessing import StandardScaler

X = df.drop('count', axis = 1)

y = df['count']

X1 = sm.add_constant(X)
import warnings

warnings.filterwarnings('ignore')

import statsmodels.api as sm

LR = sm.OLS(y, X1).fit()

LR.summary()
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,BaggingRegressor

import warnings 

from sklearn.exceptions import DataConversionWarning

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
knn = KNeighborsRegressor()

knn_params = {'n_neighbors': np.arange(3,20), 'weights': ['uniform','distance']}

GS = GridSearchCV(knn, knn_params,cv = 5, scoring = 'neg_mean_squared_error' )

GS.fit(X, y)

GS.best_params_
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sc = StandardScaler()

X_scaled = sc.fit_transform(X)

GS.fit(X_scaled, y)

GS.best_params_
dt = DecisionTreeRegressor(random_state = 0)

dt_params = {'max_depth': np.arange(3,55), 'min_samples_leaf' : np.arange(2,20) }

GS_dt = GridSearchCV(dt, dt_params,cv = 10, scoring = 'neg_mean_squared_error' )

GS_dt.fit(X_scaled, y)

GS_dt.best_params_
KNN = KNeighborsRegressor(n_neighbors = 19, weights = 'uniform')

DT = DecisionTreeRegressor(max_depth = 6, min_samples_leaf = 11, random_state = 0)

RF = RandomForestRegressor(n_estimators = 27, random_state = 0)

AB_RF = AdaBoostRegressor(base_estimator = RF, n_estimators = 100, random_state = 0)

GBoost = GradientBoostingRegressor(n_estimators = 100) 
models1 = []

models1.append(('KNNRegressor', KNN ))

models1.append(('DTRegressor', DT))

models1.append(('RFRegressor', RF))

models1.append(('ADABoostRegressor',AB_RF))

models1.append(('GradientBoostRegressor',GBoost))

from sklearn import model_selection

results = []

names = []

for name,model in models1:

    kfold = model_selection.KFold(shuffle = True, n_splits = 10,random_state = 0)

    cv_results = model_selection.cross_val_score(model, X_scaled, y, cv = kfold, scoring = 'neg_mean_squared_log_error')

    #print(cv_results)

    results.append(np.sqrt(np.abs(cv_results))) # every fold, RMSE scores... only for plotting purposes

    names.append(name)

    print("%s:%f(%f)" %(name, np.mean(np.sqrt(np.abs(cv_results))), 

                        np.std(np.sqrt(np.abs(cv_results)), ddof = 1)))
#from sklearn.ensemble import VotingRegressor

#stacked = VotingRegressor(estimators = [('GradientBoostRegressor',GBoost), ('KNNRegressor', KNN ), ('ADABoostRegressor',AB_RF)])
models = []

#models.append(('?VotingRegressor', stacked))
for name,model in models:

    kfold = model_selection.KFold(shuffle = True, n_splits = 17,random_state = 0)

    cv_results = model_selection.cross_val_score(model, X_scaled, y, cv = kfold, scoring = 'neg_mean_squared_error')

    #print(cv_results)

    results.append(np.sqrt(np.abs(cv_results))) # every fold, RMSE scores... only for plotting purposes

    names.append(name)

    print("%s:%f(%f)" %(name, np.mean(np.sqrt(np.abs(cv_results))), 

                        np.std(np.sqrt(np.abs(cv_results)), ddof = 1)))