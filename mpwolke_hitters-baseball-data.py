#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

from plotly.subplots import make_subplots

import plotly.graph_objects as go





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Data processing, metrics and modeling

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from datetime import datetime

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve

from sklearn import metrics

# Lgbm

import lightgbm as lgb

import catboost

from catboost import Pool

import xgboost as xgb



# Suppr warning

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/hitters-baseball-data/Hitters.csv')

df.head()
# Lets first handle numerical features with nan value

numerical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes!='O']

numerical_nan
df[numerical_nan].isna().sum()
## Replacing the numerical Missing Values



for feature in numerical_nan:

    ## We will replace by using median since there are outliers

    median_value=df[feature].median()

    

    df[feature].fillna(median_value,inplace=True)

    

df[numerical_nan].isnull().sum()
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
#fill in mean for floats

for c in df.columns:

    if df[c].dtype=='float16' or  df[c].dtype=='float32' or  df[c].dtype=='float64':

        df[c].fillna(df[c].mean())



#fill in -999 for categoricals

df = df.fillna(-999)

# Label Encoding

for f in df.columns:

    if df[f].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(df[f].values))

        df[f] = lbl.transform(list(df[f].values))

        

print('Labelling done.')
# Find correlations with the target and sort

correlations = df.corr()['Salary'].sort_values()



# Display correlations

print('Most Positive Correlations:\n', correlations.tail(15))

print('\nMost Negative Correlations:\n', correlations.head(15))
print(df['Salary'].skew())

print(df['Salary'].kurtosis())
import plotly.figure_factory as ff

import seaborn as sns

ax = sns.distplot(df['Salary'])

ax
import plotly.express as px

fig = px.scatter(x=df['CRuns'],y=df['Salary'])

fig.show()
#import plotly.graph_objects as go

#from plotly.subplots import make_subplots



#fig = make_subplots(rows=1, cols=5, specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}]])



#df_aux = df[['Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess', 'Compensation_and_Benefits']]

#k = 1

#for column in df_aux.columns: 

 #   fig.add_bar(y=list(df_aux[column].value_counts()), 

  #                          x=df_aux[column].value_counts().index, name=column, row=1, col=k)

   # k+=1

#fig.show()
import plotly.figure_factory as ff

fig = make_subplots(rows=1, cols=5)

df_num = df[['Salary', 'CRuns', 'HmRun', 'Runs', 'CHits']]



fig1 = ff.create_distplot([df_num['Salary']], ['Salary'])

fig2 = ff.create_distplot([df_num['CRuns']], ['CRuns'])

fig3 =  ff.create_distplot([df_num['HmRun']], ['HmRun'])

fig4 =  ff.create_distplot([df_num['Runs']], ['Runs'])

fig5 =  ff.create_distplot([df_num['CHits']], ['CHits'])



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

df_aux = df[['League', 'Division', 'NewLeague']]

f, axes = plt.subplots(1, 2)

k = 0

for column in df_aux.columns[:-1]:

    g = sns.boxplot(x=column, y='NewLeague',

                    data=df_aux, ax=axes[k])

    g.set_xticklabels(labels=g.get_xticklabels(),rotation=90)

    k +=1 

g
ax = sns.heatmap(df.corr(), annot=True, fmt=".4f")
import math

from scipy.interpolate import interp1d

df_salary = df[~df['CRuns'].isna()]

df_salary_ = df_salary[~df_salary['Salary'].isna()]

df_salary_ = df_salary_.sort_values('CRuns',  ascending=False)

interpolate_poly = interp1d(kind='linear', x=list(df_salary_['CRuns']), y=list(df_salary_['Salary']))

salaries =[]

for Salary, CRuns in zip(df_salary['Salary'], df_salary['CRuns']):

    if math.isnan(float(Salary)):

        Salary_interpolated = interpolate_poly(CRuns)

        salaries.append(Salary_interpolated)

    else:

        salaries.append(int(Salary))

df_salary['new_salary'] = salaries
df_salary = df_salary.sort_values('CRuns', ascending=False)

fig = go.Figure()

fig.add_trace(go.Scatter(x=list(df_salary['CRuns']), 

                         y=list(df_salary['Salary']), mode='markers', name='Original Salary'))



df_salary2 = df_salary[df_salary['Salary'].isna()]

fig.add_trace(go.Scatter(x=list(df_salary2['CRuns']), 

                         y=list(df_salary2['new_salary']), mode='markers', marker_color='red', name='Interpolated Salary'))



fig.show()
y = df['Salary']

df = df.drop(['Salary'], axis=1)
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

from sklearn.linear_model import Lasso 

from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



X_train, X_test, y_train, y_test = train_test_split(

    df, y, random_state=42

)
lasso_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

lasso = Lasso(random_state=42)

clf_lasso = GridSearchCV(lasso, lasso_params, cv=5, scoring='neg_mean_squared_error', n_jobs= 4, verbose = 1)

clf_lasso.fit(df, y)

print(clf_lasso.best_estimator_)

print(clf_lasso.best_score_)
param_random_tree = {"max_depth": [None],

              "max_features": [10,15, 20, 30, 43],

              "min_samples_split": [2, 3, 10,15],

              "min_samples_leaf": [1, 3, 10,15],

              "n_estimators" :[50,100,200,300,500]}



random = RandomForestRegressor(random_state=42)

clf = GridSearchCV(random, param_random_tree, cv=5,  scoring='neg_mean_squared_error',n_jobs= 4, verbose = 1)

clf.fit(df, y)

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

clf_xgb.fit(df, y)

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

clf_gbm.fit(df, y)

print(clf_gbm.best_estimator_)

print(clf_gbm.best_score_)

# (learning_rate=0.001, max_depth=6, n_estimators=50, random_state=42)
gbm = lgb.LGBMRegressor(random_state = 42, learning_rate=0.001, max_depth=6, n_estimators=50)

model = gbm.fit(df, y)

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

clf_ada.fit(df, y)

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

clf_svr.fit(df, y)

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
result = pd.DataFrame([])

result['model'] = list(scores.keys())

result['score'] = list(scores.values())

result = result.sort_values(['score'], ascending=False)

result.head(10)