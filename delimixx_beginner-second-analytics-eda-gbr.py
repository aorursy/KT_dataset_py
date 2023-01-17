import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

% matplotlib inline

from datetime import datetime

from scipy import stats

pd.options.mode.chained_assignment = None

from scipy.stats import norm, skew

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import warnings
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.describe()
print('Train Dataset Shape : {0}'.format(train.shape))

print('Test Dataset Shape : {0}'.format(test.shape))
train.dtypes
sns.boxplot(train['count'])
train = train[np.abs(train["count"]-train["count"].mean())<=(3*train["count"].std())] 
fig,ax = plt.subplots(2,1,figsize = (10,10))

sns.distplot(train['count'],ax=ax[0])

stats.probplot(train["count"], dist='norm', fit=True, plot=ax[1])

print('Skewness : {0}'.format(train['count'].skew()))

print('Kurt : {0}'.format(train['count'].kurt()))
fig,ax = plt.subplots(2,1,figsize = (10,10))

#logcount = np.log1p(train['count']).kurt()

#rootcount = np.sqrt(train['count']).kurt()

#cubiccount = np.power(train['count'],2).kurt()

#minVal = min([logcount, rootcount, cubiccount])

#if logcount == minVal:

best = 'log'

train['count_log'] = np.log1p(train['count'])

sns.distplot(train['count_log'],ax=ax[0])

stats.probplot(train["count_log"], dist='norm', fit=True, plot=ax[1])

#elif rootcount == minVal:

    #best = 'root'

    #train['count_root'] = np.sqrt(train['count'])

    #sns.distplot(train['count_root'],ax=ax[0])

    #stats.probplot(train["count_root"], dist='norm', fit=True, plot=ax[1])

#elif cubiccount == minVal:

    #best = 'cubic'

    #train['count_cubic'] = np.power(train['count'],2)

    #sns.distplot(train['count_cubic'],ax=ax[0])

    #stats.probplot(train["count_cubic"], dist='norm', fit=True, plot=ax[1])

#print('For count, the Best TF is ' + best)
train['date']  = train.datetime.apply(lambda x: x.split()[0])

train['hour'] = train.datetime.apply(lambda x: x.split()[1].split(':')[0])

train['weekday'] = train.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())

train['month'] = train.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)

train = train.drop('datetime',axis=1)
train.shape
train.dtypes
categorical = ['date','weekday','month','hour','season','holiday','workingday','weather']

numeric = ["temp","atemp","casual","registered","humidity","windspeed","count","count_log"]
for idx in categorical:

    train[idx].astype('category')
fig,axes = plt.subplots(ncols=2 ,nrows=2)

fig.set_size_inches(15,10)

sns.boxplot(data=train,x='season',y='count',ax=axes[0][0])

sns.boxplot(data=train,x='holiday',y='count',ax=axes[0][1])

sns.boxplot(data=train,x='workingday',y='count',ax=axes[1][0])

sns.boxplot(data=train,x='weather',y='count',ax=axes[1][1])



fig1,axes1 = plt.subplots()

fig1.set_size_inches(15,10)

sns.boxplot(data=train,x='hour',y='count')
plt.subplots(figsize=(15,8))

sns.heatmap(train[numeric].corr(),annot=True)
corr = train[numeric].drop('count', axis=1).corr()

corr =corr.drop('count_log', axis=1).corr() # We already examined SalePrice correlations

plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
### count,month

plt.figure(figsize=(15,8))

monthagg = pd.DataFrame(train.groupby('month')['count'].mean()).reset_index()

sns.barplot(data=monthagg, x='month',y='count').set(title = 'Month Vs Count')
### count,season,hour

plt.figure(figsize=(15,8))

houragg = pd.DataFrame(train.groupby(['hour','season'])['count'].mean()).reset_index()

sns.pointplot(data=houragg,x=houragg['hour'],y=houragg['count'],hue=houragg['season']).set(title='Hour,Season Vs Count')
### count,hour,weekday

plt.figure(figsize=(15,8))

hourweekagg = pd.DataFrame(train.groupby(['hour','weekday'])['count'].mean()).reset_index()

sns.pointplot(data=hourweekagg,x=hourweekagg['hour'],y=hourweekagg['count'],hue=hourweekagg['weekday']).set(title='Hour,Week Vs Count')
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
target = train['count']

target_log=train['count_log']

train = train.drop('count_log',axis=1)

train = train.drop('count',axis=1)

train = train.drop('atemp',axis=1)

train = train.drop('date',axis=1)

train = train.drop('casual',axis=1)

train = train.drop('registered',axis=1)

m_dum = pd.get_dummies(train['month'],prefix='m')

ho_dum = pd.get_dummies(train['hour'],prefix='ho')

s_dum = pd.get_dummies(train['season'],prefix='s')

we_dum = pd.get_dummies(train['weather'],prefix='we')

train = pd.concat([train,s_dum,we_dum,m_dum,ho_dum],axis=1)



testid = test['datetime']

test['date']  = test.datetime.apply(lambda x: x.split()[0])

test['hour'] = test.datetime.apply(lambda x: x.split()[1].split(':')[0])

test['weekday'] = test.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())

test['month'] = test.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)

test = test.drop('datetime',axis=1)

test = test.drop('atemp',axis=1)

test = test.drop('date',axis=1)

s_dum = pd.get_dummies(test['season'],prefix='s')

we_dum = pd.get_dummies(test['weather'],prefix='we')

m_dum = pd.get_dummies(test['month'],prefix='m')

ho_dum = pd.get_dummies(test['hour'],prefix='ho')

test= pd.concat([test,s_dum,we_dum,m_dum,ho_dum],axis=1)
train.shape
test.shape
gbr = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01, max_depth=4).fit(train.values, target_log)
def loss_func(truth, prediction):

    y = np.expm1(truth)

    y_ = np.expm1(prediction)

    log1 = np.array([np.log(x + 1) for x in truth])

    log2 = np.array([np.log(x + 1) for x in prediction])

    return np.sqrt(np.mean((log1 - log2)**2))
#from sklearn.model_selection import GridSearchCV

#from sklearn.ensemble import GradientBoostingRegressor

#from sklearn.metrics import make_scorer

#param_grid = {

#    'learning_rate': [0.1, 0.01, 0.001],

#    'n_estimators': [100, 1000, 1500, 2000, 4000],

#    'max_depth': [1, 2, 3, 4, 5]

#}

#scorer = make_scorer(loss_func, greater_is_better=False)

#model = GradientBoostingRegressor(random_state=42)

#result = GridSearchCV(model, param_grid, cv=4, scoring=scorer, n_jobs=3).fit(train.values, target_log)

#print('\tParams:', result.best_params_)

#print('\tScore:', result.best_score_)
##	Params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1500}

#	Score: -0.12669018059776296
model_gbr = GradientBoostingRegressor(n_estimators=1500,max_depth=5,learning_rate=0.01).fit(train.values,target_log)
prediction = model_gbr.predict(test.values)

prediction = np.expm1(prediction)
output = pd.DataFrame()

output['datetime'] = testid

output['count'] = prediction

output.to_csv('output.csv',index=False)