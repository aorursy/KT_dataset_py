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
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

!pip3 install catboost
!pip3 install xgboost 
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/hackerearth-employee-attrition/Train.csv')
df_test = pd.read_csv('/kaggle/input/hackerearth-employee-attrition/Test.csv')
df_test.head()
import seaborn as sns
sns.distplot(df['Attrition_rate'])

print('skew',df['Attrition_rate'].skew())
print('kurtosis',df['Attrition_rate'].kurtosis())
import plotly.express as px
fig = px.pie(values=df['Gender'].value_counts(), names=df['Gender'].value_counts().index, title='Gernes')
fig.show()
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=5, specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}]])

df_aux = df[['Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess', 'Compensation_and_Benefits']]
k = 1
for column in df_aux.columns: 
    fig.add_bar(y=list(df_aux[column].value_counts()), 
                            x=df_aux[column].value_counts().index, name=column, row=1, col=k)
    k+=1
fig.show()
import plotly.figure_factory as ff
fig = make_subplots(rows=1, cols=5)
df_num = df[['Time_since_promotion', 'growth_rate', 'Travel_Rate', 'Post_Level', 'Education_Level']]

fig1 = ff.create_distplot([df_num['Time_since_promotion']], ['Time_since_promotion'])
fig2 = ff.create_distplot([df_num['growth_rate']], ['growth_rate'])
fig3 =  ff.create_distplot([df_num['Travel_Rate']], ['Travel_Rate'])
fig4 =  ff.create_distplot([df_num['Post_Level']], ['Post_Level'])
fig5 =  ff.create_distplot([df_num['Education_Level']], ['Education_Level'])

fig.add_trace(go.Histogram(fig1['data'][0], marker_color='blue'), row=1, col=1)
fig.add_trace(go.Histogram(fig2['data'][0],marker_color='red'), row=1, col=2)
fig.add_trace(go.Histogram(fig3['data'][0], marker_color='green'), row=1, col=3)
fig.add_trace(go.Histogram(fig4['data'][0],marker_color='yellow'), row=1, col=4)
fig.add_trace(go.Histogram(fig5['data'][0],marker_color='purple'), row=1, col=5)


fig.show()
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 20.7,5.27
df_aux = df[['Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess', 'Compensation_and_Benefits', 'Attrition_rate']]
f, axes = plt.subplots(1, 5)
k = 0
for column in df_aux.columns[:-1]:
    g = sns.boxplot(x=column, y='Attrition_rate',
                    data=df_aux, ax=axes[k])
    g.set_xticklabels(labels=g.get_xticklabels(),rotation=90)
    k +=1 
g
fig = px.bar(x=df.isna().sum().index, y=df.isna().sum())
fig.show()
ax = sns.heatmap(df.corr(), annot=True, fmt=".4f")
import math
from scipy.interpolate import interp1d
df_age = df[~df['Time_of_service'].isna()]
df_age_ = df_age[~df_age['Age'].isna()]
df_age_ = df_age_.sort_values('Time_of_service',  ascending=False)
interpolate_poly = interp1d(kind='linear', x=list(df_age_['Time_of_service']), y=list(df_age_['Age']))
ages =[]
for age, time_service in zip(df_age['Age'], df_age['Time_of_service']):
    if math.isnan(float(age)):
        age_interpolated = interpolate_poly(time_service)
        ages.append(age_interpolated)
    else:
        ages.append(int(age))
df_age['new_age'] = ages
df_age = df_age.sort_values('Time_of_service', ascending=False)
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(df_age['Time_of_service']), 
                         y=list(df_age['Age']), mode='markers', name='Original Age'))

df_age2 = df_age[df_age['Age'].isna()]
fig.add_trace(go.Scatter(x=list(df_age2['Time_of_service']), 
                         y=list(df_age2['new_age']), mode='markers', marker_color='red', name='Interpolated Age'))

fig.show()
pay = []
work = []
pp = df_age['Pay_Scale'].mode()
ww = df_age['Work_Life_balance'].mode()
for p, w in zip(df_age['Pay_Scale'], df_age['Work_Life_balance']):
    if math.isnan(float(p)):
        pay.append(pp)
    else:
        pay.append(p)
    if math.isnan(float(w)):
        work.append(ww)
    else:
        work.append(w)

df_age['Pay_Scale'] = pay
df_age['Work_Life_balance'] = work
df_age.info()
rel_status = pd.get_dummies(df_age['Relationship_Status'])
hometown = pd.get_dummies(df_age['Hometown'])
unit = pd.get_dummies(df_age['Unit'])
decision = pd.get_dummies(df_age['Decision_skill_possess'])
compenssion = pd.get_dummies(df_age['Compensation_and_Benefits'])
to_work = df_age[['Education_Level', 'Time_of_service', 'Time_since_promotion', 'growth_rate', 'Travel_Rate', 'Post_Level', 'Pay_Scale', 'Work_Life_balance', 'VAR1', 'VAR3', 'VAR5', 'VAR6','VAR7', 'Attrition_rate', 'new_age']]
df_to_modelling = pd.concat([to_work, compenssion, decision, unit, hometown, rel_status], axis=1)
y = df_to_modelling['Attrition_rate']
df_to_modelling = df_to_modelling.drop(['Attrition_rate'], axis=1)
df_to_modelling['Pay_Scale'] = df_to_modelling['Pay_Scale'].astype(float)
df_to_modelling['new_age'] = df_to_modelling['new_age'].astype(int)
df_to_modelling['Work_Life_balance'] = df_to_modelling['Work_Life_balance'].astype(float)
def plot_predict(pred, true):
    indexs = []
    for i in range(len(pred)):
        indexs.append(i)
        

    fig = go.Figure()

    fig.add_trace(go.Line(
        x=indexs,
        y=pred,
        name="Predict"
    ))

    fig.add_trace(go.Line(
        x=indexs,
        y=true,
        name="Test"
    ))

    fig.show()
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(
    df_to_modelling, y, random_state=42
)

param_random_tree = {"max_depth": [None],
              "max_features": [10,15, 20, 30, 43],
              "min_samples_split": [2, 3, 10,15],
              "min_samples_leaf": [1, 3, 10,15],
              "n_estimators" :[50,100,200,300,500]}

random = RandomForestRegressor(random_state=42)
clf = GridSearchCV(random, param_random_tree, cv=5,  scoring='neg_mean_squared_error',n_jobs= 4, verbose = 1)
clf.fit(X_train, y_train)
print(clf.best_estimator_)
print(clf.best_score_)
# (max_features=10, min_samples_leaf=15, n_estimators=500, random_state=42)

scores = {}
random = RandomForestRegressor(max_features=10, min_samples_leaf=15, n_estimators=500, random_state=42)
model = random.fit(X_train, y_train)
pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))
score = 100* max(0, 1-mean_squared_error(y_test, pred))
print(score)
scores['RF'] = score
import xgboost
xgboost_params = {'max_features': [10,15, 20, 30],
                  'n_estimators' :[25,50,100],
                   'learning_rate': [0.0001, 0.001, 0.01, 0.1],
                  'gamma':[0.5, 0.1, 1, 10],
                  'max_depth':[5, 10, 15]}

xgb = xgboost.XGBRegressor(random_state=42)
clf_xgb = GridSearchCV(xgb, xgboost_params, cv=5,  scoring='neg_mean_squared_error',n_jobs= 4, verbose = 1)
clf_xgb.fit(df_to_modelling, y)
print(clf_xgb.best_estimator_)
print(clf_xgb.best_score_)
"""
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=1, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=5, max_features=10,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=42,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
"""


import xgboost
xgb = xgboost.XGBRegressor(gamma=1, random_state=42, max_depth=5, max_features=10,learning_rate=0.1, n_estimators=100)
model = xgb.fit(X_train, y_train)
pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))
score = 100* max(0, 1-mean_squared_error(y_test, pred))
print(score)
scores['XGB'] = score
import lightgbm as lgb
lightgbm_params ={'learning_rate':[0.0001, 0.001, 0.003, 0.01, 0.1],
                  'n_estimators':[10,20, 50, 100],
                 'max_depth':[4, 6, 10, 15, 20, 50]}
gbm = lgb.LGBMRegressor(random_state = 42)
clf_gbm = GridSearchCV(gbm, lightgbm_params, cv=5,  scoring='neg_mean_squared_error',n_jobs= 4, verbose = 1)
clf_gbm.fit(df_to_modelling, y)
print(clf_gbm.best_estimator_)
print(clf_gbm.best_score_)
# (learning_rate=0.001, max_depth=6, n_estimators=50, random_state=42)
gbm = lgb.LGBMRegressor(random_state = 42, learning_rate=0.001, max_depth=6, n_estimators=50)
model = gbm.fit(df_to_modelling, y)
pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))
score = 100* max(0, 1-mean_squared_error(y_test, pred))
print(score)
scores['LGBM'] = score
from sklearn.ensemble import AdaBoostRegressor
adam_boosting_params = {'learning_rate':[0.0001, 0.001, 0.003, 0.01, 0.1,1],
                        'n_estimators':[10,20, 50, 100]}
ada = AdaBoostRegressor(random_state=42)
clf_ada = GridSearchCV(ada, adam_boosting_params, cv=5,  scoring='neg_mean_squared_error',n_jobs= 4, verbose = 1)
clf_ada.fit(df_to_modelling, y)
print(clf_ada.best_estimator_)
print(clf_ada.best_score_)
# (learning_rate=0.0001, n_estimators=100, random_state=42)
ada = AdaBoostRegressor(random_state=42, learning_rate=0.0001, n_estimators=100)
model = ada.fit(X_train, y_train)
pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))
score = 100* max(0, 1-mean_squared_error(y_test, pred))
print(score)
scores['ADA'] = score
from sklearn.svm import LinearSVR

svr_params = {'C':[0.0001, 0.001,0.01, 0.1, 1 , 10, 100]}
svr = LinearSVR(random_state=42)
clf_svr = GridSearchCV(svr, svr_params, cv=5, scoring='neg_mean_squared_error', n_jobs=4, verbose=1)
clf_svr.fit(df_to_modelling, y)
print(clf_svr.best_estimator_)
print(clf_svr.best_score_)
# (C=0.001, random_state=42)
lvr = LinearSVR(C=0.001, random_state=42)
model = svr.fit(X_train, y_train)
pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))
score = 100* max(0, 1-mean_squared_error(y_test, pred))
print(score)
scores['SVR'] = score
estimators  =  [
    ('rf', RandomForestRegressor(max_features=10, min_samples_leaf=15, n_estimators=500, random_state=42)),
    ('svr', LinearSVR(C=0.001, random_state=42)),
    ('ada', AdaBoostRegressor(random_state=42, learning_rate=0.0001, n_estimators=100)),
    ('lgb', lgb.LGBMRegressor(random_state = 42, learning_rate=0.001, max_depth=6, n_estimators=50)),
    
]
clf = StackingRegressor(
    estimators=estimators, final_estimator=xgboost.XGBRegressor(gamma=1, random_state=42, max_depth=5, max_features=10,learning_rate=0.1, n_estimators=100)
)
model = clf.fit(X_train, y_train)
pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))
score = 100* max(0, 1-mean_squared_error(y_test, pred))
print(score)
scores['STACK'] = score
from sklearn.ensemble import VotingRegressor
estimators  =  [
    ('rf', RandomForestRegressor(max_features=10, min_samples_leaf=15, n_estimators=500, random_state=42)),
    ('svr', LinearSVR(C=0.001, random_state=42)),
    ('ada', AdaBoostRegressor(random_state=42, learning_rate=0.0001, n_estimators=100)),
    ('lgb', lgb.LGBMRegressor(random_state = 42, learning_rate=0.001, max_depth=6, n_estimators=50)),
    ('xgb', xgboost.XGBRegressor(gamma=1, random_state=42, max_depth=5, max_features=10,learning_rate=0.1, n_estimators=100))
]
clf = VotingRegressor(
    estimators=estimators
)
model = clf.fit(X_train, y_train)
pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))
score = 100* max(0, 1-mean_squared_error(y_test, pred))
print(score)
scores['VOLTING'] = score
result = pd.DataFrame([])
result['model'] = list(scores.keys())
result['score'] = list(scores.values())
result = result.sort_values(['score'], ascending=False)
result.head(10)