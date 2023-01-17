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

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore")
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")

test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")
train.head()
test.head()
train.info()
test.info()
train.isnull().sum()
target = train['count']
from scipy import stats

from scipy.stats import norm
sns.distplot(train['count'],fit=norm)
stats.probplot(train['count'], plot = plt)
train["log_count"] = np.log(target+1)
sns.distplot(train["log_count"], fit=norm)
stats.probplot(train["log_count"], plot = plt)
feature_names=list(test)

df_train=train[feature_names]

df=pd.concat((df_train, test))
print(train.shape, test.shape, df.shape)
df.head()
df.info()
import datetime
tmp = pd.to_datetime(train['datetime'])
df['datetime'] = pd.to_datetime(df['datetime'])

df['day'] = df['datetime'].dt.day

df['hour'] = df['datetime'].dt.hour

df['dayofweek'] = df['datetime'].dt.dayofweek

df['month'] = df['datetime'].dt.month

df['year'] = df['datetime'].dt.year

df['weekend'] = (df['dayofweek'] ==5) | (df['dayofweek'] == 6)
train['datetime'] = pd.to_datetime(train['datetime'])

train['day'] = train['datetime'].dt.day

train['hour'] = train['datetime'].dt.hour

train['dayofweek'] = train['datetime'].dt.dayofweek

train['month'] = train['datetime'].dt.month

train['year'] = train['datetime'].dt.year

train['weekend'] = (train['dayofweek'] ==5) | (train['dayofweek'] == 6)
df.drop(['datetime'], axis=1, inplace=True)
figure, axs = plt.subplots(3,2, figsize = (15,10))



sns.barplot(data=train, x = "day", y = target, ax = axs[0][0])

sns.barplot(data=train, x = "hour", y = target, ax = axs[0][1])

sns.barplot(data=train, x = "dayofweek", y = target, ax = axs[1][0])

sns.barplot(data=train, x = "weekend", y = target, ax = axs[1][1])

sns.barplot(data=train, x = "month", y = target, ax = axs[2][0])

sns.barplot(data=train, x = "year", y = target, ax = axs[2][1])
df=df.drop(columns=['month', 'day'])
df
sns.barplot(data=df[:len(train)], x='season', y=target)
season_encoded = pd.get_dummies(df['season'],prefix= 'season')

df = pd.concat((df,season_encoded), axis=1)

df = df.drop(columns = 'season')
sns.barplot(data=df[:len(train)], x='holiday', y=target)
df['holiday'] = df['holiday']
sns.barplot(data=df[:len(train)], x='workingday', y=target)
df['workingday'] = df['workingday']
sns.barplot(data=df[:len(train)], x='weather', y=target)

df['weather'] = df['weather']
weather_encoded = pd.get_dummies(df['weather'],prefix= 'weather')

df = pd.concat((df,weather_encoded), axis=1)

df = df.drop(columns = 'weather')
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows = 5)

fig.set_size_inches(20,30)



sns.pointplot(data = train, x = "hour", y = "count", ax = ax1)

sns.pointplot(data = train, x = "hour", y = "count", hue = "season", ax = ax2)

sns.pointplot(data = train, x = "hour", y = "count", hue = "holiday", ax = ax3)

sns.pointplot(data = train, x = "hour", y = "count", hue = "workingday", ax = ax4)

sns.pointplot(data = train, x = "hour", y = "count", hue = "weather",  ax = ax5)
from scipy.stats import skew

skew = df.apply(lambda x: skew(x))

skew.sort_values(ascending = False)
skew = skew[abs(skew) > 0.5]

skew
cor = train.iloc[:,1:-1].corr()

cor.head()
mask = np.array(cor)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sns.heatmap(cor,mask= mask,square=True,annot=True)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = df.iloc[:,:6]

vif_data.info()
vif = pd.DataFrame()

vif['Features'] = vif_data.columns

vif['vif'] = [variance_inflation_factor(

             vif_data.values, i) for i in range(vif_data.shape[1])]

vif.sort_values(by='vif',ascending=False)
from sklearn.decomposition import PCA

pca=PCA(n_components=1)

pca.fit(df[['temp', 'atemp']])
pca.explained_variance_ratio_
df['pca']=pca.fit_transform(df[['temp','atemp']])
sns.distplot(df['pca'], fit=norm)
fig, [ax1,ax2,ax3] = plt.subplots(1,3)

fig.set_size_inches(12,5)

sns.regplot(train['temp'], 'count', data = train, ax=ax1)

sns.regplot(train['humidity'], 'count', data = train, ax=ax2)

sns.regplot(train['windspeed'], 'count', data = train, ax=ax3)
stats.pearsonr(train['temp'],target)
sns.countplot(data = df, x = "windspeed")
df.loc[df['windspeed']==0, 'windspeed']=df['windspeed'].mean()
df = df.drop(columns=['temp','atemp'])
fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (15,20))

sns.boxplot(data = train, y="count", x = "holiday", orient = "v", ax = axes[0][0])

sns.boxplot(data = train, y="count", x = "workingday", orient = "v", ax = axes[0][1])

sns.boxplot(data = train, y="count", x = "hour", orient = "v", ax = axes[1][0])

sns.boxplot(data = train, y="count", x = "dayofweek", orient = "v", ax = axes[1][1])

sns.boxplot(data = train, y="count", x = "year", orient = "v", ax = axes[2][0])
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
new_train = df[:train.shape[0]]

new_test = df[train.shape[0]:]
target = train['log_count']
X_train, X_val, y_train, y_val = train_test_split(new_train, target, test_size=0.2, shuffle=True)
def rmsle_score(preds, true):

    rmsle_score = (np.sum((np.log1p(preds)-np.log1p(true))**2)/len(true))**0.5

    return rmsle_score
from sklearn.metrics.scorer import make_scorer



RMSLE = make_scorer(rmsle_score)
import statsmodels.api as sm
model = sm.OLS(target.values, new_train.astype(float))
re = model.fit()
re.summary()
param = {'alpha':[1e-03,0.01,0.1,0.5,0.8,1], 'normalize':[True,False], 'tol':[1e-06,1e-05,5e-05,1e-04,5e-04,1e-03]}
lasso = make_pipeline(GridSearchCV(Lasso(random_state=1),param,

                                  cv=10, scoring = RMSLE))
lasso.fit(X_train,y_train)
la_yhat = lasso.predict(X_val)
s_lasso = rmsle_score(la_yhat,y_val)

s_lasso
pred_la = lasso.predict(new_test)
param_e = {'alpha' :[0.1,1.0,10], 'max_iter' :[1000000], 'l1_ratio':[0.04,0.05], 

           'normalize':[True,False]}
Enet = make_pipeline(GridSearchCV(ElasticNet(random_state=1),param_e,

                     cv=10, scoring = RMSLE))
Enet.fit(X_train,y_train)
Enet_yhat = Enet.predict(X_val)
s_Enet = rmsle_score(Enet_yhat,y_val)

s_Enet
pred_Enet = Enet.predict(new_test)
param_Rf =  {'min_samples_split' : [3,4,6,10], 'n_estimators' : [70,100], 'random_state': [5] }
RF = make_pipeline(GridSearchCV(RandomForestRegressor(random_state=1),param_Rf,

                   cv=10, scoring = RMSLE))
RF.fit(X_train,y_train)
RF_yhat = RF.predict(X_val)

s_RF = rmsle_score(RF_yhat,y_val)

s_RF
pred_RF = RF.predict(new_test)
param_GB = [{'learning_rate': [1,0.1,0.01,0.001],

              'n_estimators': [50, 100, 200, 500, 1000]}]
GB = make_pipeline(GridSearchCV(GradientBoostingRegressor(random_state=1),param_GB,

                   cv=10, scoring = RMSLE))
GB.fit(X_train,y_train)
GB_yhat = GB.predict(X_val)

s_GB = rmsle_score(GB_yhat,y_val)

s_GB
pred_GB = GB.predict(new_test)
param_lgb = param_grid = [{

    'n_estimators': [400, 700, 1000], 

    'max_depth': [15,20,25],

    'num_leaves': [50, 100, 200],

    'min_split_gain': [0.3, 0.4],

}]
lgb = make_pipeline(GridSearchCV(LGBMRegressor(verbose_eval=False,random_state=1),param_lgb,

                    cv=10, scoring = RMSLE))
lgb.fit(X_train,y_train)
lgb_yhat = lgb.predict(X_val)

s_lgb = rmsle_score(lgb_yhat,y_val)

s_lgb
pred_lgb = lgb.predict(new_test)
list_scores = [s_lasso, s_Enet, s_RF,s_GB,s_lgb]

list_regressors = ['Lasso','Enet','RF','GB','lgb']
sns.barplot(x=list_regressors, y=list_scores)

plt.ylabel('RMSE')
predictions = {'Lasso': pred_la,

               'ElaNet': pred_Enet, 

               'RF': pred_RF,

               'GB': pred_GB,

               #'XGB' : pred_xgb,

               'lgb' : pred_lgb

              }
df_predictions = pd.DataFrame(data=predictions) 

df_predictions.corr()
plt.figure(figsize=(7, 7))

sns.heatmap(df_predictions.corr(),linewidths=1.5,

            annot=True, 

            square=True,          

            yticklabels=df_predictions.columns , 

            xticklabels=df_predictions.columns)

RF.fit(new_train,target)
log_pred=RF.predict(new_test)

predictions=np.exp(log_pred)-1
sub = pd.DataFrame()

sub['datetime'] = test['datetime']

sub['count'] = predictions

sub.head()
sub.to_csv('submission.csv', index=False)
lgb.fit(new_train,target)
log_pred_lgb=lgb.predict(new_test)

predictions_lgb=np.exp(log_pred_lgb)-1
sub = pd.DataFrame()

sub['datetime'] = test['datetime']

sub['count'] = predictions_lgb

sub.head()
ensemble = (0.6*predictions + 0.4*predictions_lgb) 
sub = pd.DataFrame()

sub['datetime'] = test['datetime']

sub['count'] = ensemble

sub.head()
sub.to_csv('submission.csv', index=False)