%pip install -U pip 

%pip install scikit-learn==0.20.3
!pip list |grep scikit-learn
import sklearn

print(sklearn.__version__)
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

import matplotlib.pyplot as plt

import seaborn as sns

import sys

import warnings

import pandas_profiling as pp
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.express as px

init_notebook_mode(connected=True)
pd.set_option('display.width', 500)

pd.set_option('display.max_columns', 30)
warnings.filterwarnings("ignore")
df_train = pd.read_csv(dirname+'/train.csv')

df_test = pd.read_csv(dirname+'/test.csv')
df_train.head()
df_test.head()
pp.ProfileReport(df_train)
pp.ProfileReport(df_test)
df_train.isna().sum()
df_train['datetime'] = pd.to_datetime(df_train['datetime'])

df_train['hour'] = df_train['datetime'].map(lambda x :x.hour)

df_train['day'] = df_train['datetime'].map(lambda x:x.day)

df_train['year'] = df_train['datetime'].map(lambda x:x.year)

df_train['month'] = df_train['datetime'].map(lambda x:x.month)

df_train['day_of_week'] = df_train['datetime'].map(lambda x:x.dayofweek)
df_train.head()
df_test['datetime'] = pd.to_datetime(df_test['datetime'])

df_test['hour'] = df_test['datetime'].map(lambda x :x.hour)

df_test['day'] = df_test['datetime'].map(lambda x:x.day)

df_test['year'] = df_test['datetime'].map(lambda x:x.year)

df_test['month'] = df_test['datetime'].map(lambda x:x.month)

df_test['day_of_week'] = df_test['datetime'].map(lambda x:x.dayofweek)
df_test.head()
for col in ['casual', 'registered', 'cnt']:

    df_train['%s_log' % col] = np.log1p(df_train[col])
df_train['date'] = df_train['datetime'].apply(lambda x:x.date())
eda_date = pd.DataFrame(df_train.groupby('date').sum()['cnt']).reset_index()

eda_date.head()
fig = px.line(eda_date, x='date', y='cnt')

fig.show()
eda_date = pd.DataFrame(df_train.groupby(['weekday','month','year']).mean()['cnt']).reset_index()

eda_date.head()
fig = px.bar(eda_date, x='weekday', y='cnt', facet_col='month', facet_row='year')

fig.show()
eda_date = pd.DataFrame(df_train.groupby(['hour','holiday', 'workingday']).mean()['cnt']).reset_index().dropna()

eda_date.head()
fig = px.bar(eda_date, x='hour', y='cnt', facet_col='holiday', facet_row='workingday')

fig.show()
eda_date = pd.DataFrame(df_train.groupby(['hour','weekday','month']).mean()['cnt']).reset_index()

eda_date.head()
fig = px.bar(eda_date, x='hour', y='cnt', facet_col='weekday',facet_row='month')

fig.show()
eda_date = pd.DataFrame(df_train.groupby(['day','month','holiday','year']).sum()['cnt']).reset_index().dropna()

eda_date.head()
fig = px.bar(eda_date, x='day', y='cnt', facet_col='month', facet_row='year', color='holiday')

fig.show()
eda_date = pd.DataFrame(df_train.groupby(['hour','weather', 'weekday']).mean()['cnt']).reset_index().dropna()

eda_date.head()
fig = px.bar(eda_date, x='hour', y='cnt', facet_col='weekday', facet_row='weather')

fig.show()
eda_date = pd.DataFrame(df_train.groupby(['hour','weather', 'workingday']).mean()['cnt']).reset_index().dropna()

eda_date.head()
fig = px.bar(eda_date, x='hour', y='cnt', facet_col='weather', facet_row='workingday')

fig.show()
eda_date = pd.DataFrame(df_train.groupby(['hour','weekday','season']).mean()['cnt']).reset_index()

eda_date.head()
fig = px.bar(eda_date, x='hour', y='cnt', facet_col='weekday',facet_row='season')

fig.show()
sns.distplot(df_train['temp'])
eda_num = pd.DataFrame(df_train.groupby(['temp']).mean()['cnt']).reset_index().dropna()

eda_num.head()
fig = px.scatter(eda_num,x='temp',y='cnt')

fig.show()
sns.distplot(df_train['atemp'])
eda_num = pd.DataFrame(df_train.groupby(['atemp']).mean()['cnt']).reset_index().dropna()

eda_num.head()
fig = px.scatter(eda_num,x='atemp',y='cnt')

fig.show()
eda_num = pd.DataFrame(df_train.groupby(['temp', 'weekday']).mean()['cnt']).reset_index().dropna()

eda_num.head()
fig = px.scatter(eda_num, x='temp', y='cnt', facet_row='weekday')

fig.show()
sns.distplot(df_train['humidity'])
eda_num = pd.DataFrame(df_train.groupby(['humidity']).mean()['cnt']).reset_index().dropna()

eda_num.head()
fig = px.scatter(eda_num,x='humidity',y='cnt')

fig.show()
eda_num = pd.DataFrame(df_train.groupby(['humidity','weekday']).mean()['cnt']).reset_index().dropna()

eda_num.head()
fig = px.scatter(eda_num, x='humidity', y='cnt', facet_row='weekday')

fig.show()
fig = px.box(df_train,y='cnt',x='hour',color='weekday')

fig.show()
sns.distplot(df_train['windspeed'],hist=False)
eda_num = pd.DataFrame(df_train.groupby(['windspeed']).mean()['cnt']).reset_index().dropna()

eda_num.head()
fig = px.scatter(eda_num,x='windspeed',y='cnt')

fig.show()
# df_train['working_hour'] = df_train['hour'].apply(lambda x: x >=9 and x <17)

# df_train['night'] = df_train['hour'].apply(lambda x: x  <=6)

# df_train['year'] = df_train['year'].apply(lambda x: x  ==2012)

# df.drop(['date','datetime'])
df_train.head()
# Metrics

from sklearn.metrics import mean_squared_error



# Model Selection

from sklearn.model_selection import KFold, cross_val_score



# Model

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
X_train = df_train[['weather', 'temp', 'atemp', 'humidity', 'windspeed', 

                    'holiday', 'workingday', 'season', 'hour', 'day_of_week', 'year']]

y_train_cas = df_train['casual_log']

y_train_reg = df_train['registered_log']
X_test = df_test[['weather', 'temp', 'atemp', 'humidity', 'windspeed', 

    'holiday', 'workingday', 'season', 'hour', 'day_of_week', 'year']]
rf_train = df_train[['weather', 'temp', 'atemp', 'windspeed',

    'workingday', 'season', 'holiday',

    'hour', 'weekday']]



rf_test = df_test[['weather', 'temp', 'atemp', 'windspeed',

    'workingday', 'season', 'holiday',

    'hour', 'weekday']]
kf = KFold(5)
params = {'n_estimators': 150, 'max_depth': 5, 'random_state': 0, 'min_samples_leaf' : 10, 'learning_rate': 0.1, 'subsample': 0.7, 'loss': 'ls'}

gb_cas = GradientBoostingRegressor(**params)

gb_reg = GradientBoostingRegressor(**params)
score_cas = cross_val_score(gb_cas,X_train,y_train_cas,cv=5,scoring='neg_mean_squared_error')

print(f'Casual CV Score: {score_cas.mean()} ± {score_cas.std()}')
score_reg = cross_val_score(gb_reg,X_train,y_train_reg,cv=5,scoring='neg_mean_squared_error')

print(f'CV Score: {score_reg.mean()} ± {score_reg.std()}')
rf_cas = RandomForestRegressor(n_estimators = 1000, 

    min_samples_split = 12, 

    n_jobs = -1,

    random_state = 123, )

rf_reg = RandomForestRegressor(n_estimators = 1000, 

    min_samples_split = 12, 

    n_jobs = -1,

    random_state = 123, )
score_cas = cross_val_score(rf_cas,rf_train,y_train_cas,cv=5,scoring='neg_mean_squared_error')

print(f'Casual CV Score: {score_cas.mean()} ± {score_cas.std()}')
score_reg = cross_val_score(rf_reg,rf_train,y_train_reg,cv=5,scoring='neg_mean_squared_error')

print(f'CV Score: {score_reg.mean()} ± {score_reg.std()}')
gb_cas.fit(X_train,y_train_cas)

gb_reg.fit(X_train,y_train_reg)



y_pred_cas = gb_cas.predict(X_test)

y_pred_cas = np.exp(y_pred_cas) - 1





y_pred_reg = gb_reg.predict(X_test)

y_pred_reg = np.exp(y_pred_reg) - 1
# df_test['count'] = y_pred_cas + y_pred_reg

y_gb = y_pred_cas + y_pred_reg



y_gb[:20]
# params = {'n_estimators': 150, 'max_depth': 5, 'random_state': 0, 'min_samples_leaf' : 10, 'learning_rate': 0.1, 'subsample': 0.7, 'loss': 'ls'}

# model = GradientBoostingRegressor(**params)

rf_cas.fit(rf_train,y_train_cas)

rf_reg.fit(rf_train,y_train_reg)



y_pred_cas = rf_cas.predict(rf_test)

y_pred_cas = np.exp(y_pred_cas) - 1



y_pred_reg = rf_reg.predict(rf_test)

y_pred_reg = np.exp(y_pred_reg) - 1





y_rf = y_pred_cas + y_pred_reg



y_rf[:20]

y_voting = .2*y_rf + .8*y_gb
y_voting[:20]
df_test['count'] = y_voting
df_test['count'].head(10)
# import shap

# import lime
# explainerSKGBT_cas = shap.TreeExplainer(gb_cas)

# shap_values_SKGBT_test_cas = explainerSKGBT_cas.shap_values(X_test)

# shap_values_SKGBT_train_cas = explainerSKGBT_cas.shap_values(X_train)
# # Scikit GBT

# df_shap_SKGBT_test_cas = pd.DataFrame(shap_values_SKGBT_test_cas, columns=X_test.columns.values)

# df_shap_SKGBT_train_cas = pd.DataFrame(shap_values_SKGBT_train_cas, columns=X_train.columns.values)
# # if a feature has 10 or less unique values then treat it as categorical

# categorical_features = np.argwhere(np.array([len(set(X_train.values[:,x]))

# for x in range(X_train.values.shape[1])]) <= 10).flatten()

 

# # LIME has one explainer for all models

# explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,

# feature_names=X_train.columns.values.tolist(),

# class_names=['price'],

# categorical_features=categorical_features,

# verbose=True,

#                                                    mode='regression')
# shap.initjs()

# j=30

# shap.force_plot(explainerSKGBT_cas.expected_value, shap_values_SKGBT_test_cas[j], X_test.iloc[[j]])
# expSKGBT = explainer.explain_instance(X_test.values[j], gb_cas.predict, num_features=5)

# expSKGBT.show_in_notebook(show_table=True)
# shap.initjs()

# j=1000

# shap.force_plot(explainerSKGBT_cas.expected_value, shap_values_SKGBT_test_cas[j], X_test.iloc[[j]])
# expSKGBT = explainer.explain_instance(X_test.values[j], gb_cas.predict, num_features=5)

# expSKGBT.show_in_notebook(show_table=True)
# explainerSKGBT_reg = shap.TreeExplainer(gb_reg)

# shap_values_SKGBT_test_reg = explainerSKGBT_reg.shap_values(X_test)

# shap_values_SKGBT_train_reg = explainerSKGBT_reg.shap_values(X_train)
# Scikit GBT

# df_shap_SKGBT_test_reg = pd.DataFrame(shap_values_SKGBT_test_reg, columns=X_test.columns.values)

# df_shap_SKGBT_train_reg = pd.DataFrame(shap_values_SKGBT_train_reg, columns=X_train.columns.values)
# shap.initjs()

# j=30

# shap.force_plot(explainerSKGBT_reg.expected_value, shap_values_SKGBT_test_reg[j], X_test.iloc[[j]])
# expSKGBT = explainer.explain_instance(X_test.values[j], gb_reg.predict, num_features=5)

# expSKGBT.show_in_notebook(show_table=True)
# j=1000

# shap.force_plot(explainerSKGBT_reg.expected_value, shap_values_SKGBT_test_reg[j], X_test.iloc[[j]])
# expSKGBT = explainer.explain_instance(X_test.values[j], gb_reg.predict, num_features=5)

# expSKGBT.show_in_notebook(show_table=True)


submission = pd.read_csv(dirname+"/sampleSubmission.csv")



submission['count'] = df_test["count"]



submission.head(10)



submission.to_csv('submission35.csv',index=False)