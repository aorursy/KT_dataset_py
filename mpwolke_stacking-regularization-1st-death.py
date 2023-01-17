# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



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
df = pd.read_csv('../input/hackathon/task_2-COVID-19-death_cases_per_country_after_frist_death-till_26_June.csv')

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
# categorical features with missing values

categorical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes=='O']

print(categorical_nan)
# replacing missing values in categorical features

for feature in categorical_nan:

    df[feature] = df[feature].fillna('None')
df[categorical_nan].isna().sum()
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
print(df['date_first_death'].skew())

print(df['date_first_death'].kurtosis())
import plotly.figure_factory as ff

import seaborn as sns

ax = sns.distplot(df['date_first_death'])

ax
import plotly.express as px

fig = px.scatter(x=df['country_name'],y=df['date_first_death'])

fig.show()
import collections



dict_ = {}

for alpha_3_code in df['alpha_3_code'].unique():

    tmp = df[df['alpha_3_code']==alpha_3_code]

    dict_[alpha_3_code] = tmp['date_first_death'].mean()

means = []

ordered = collections.OrderedDict(sorted(dict_.items()))

for alpha_3_code, value in ordered.items():

    means.append(value)
test = pd.DataFrame([ordered]) 
fig = px.line(x=sorted(df['alpha_3_code'].unique()), y=means, title='Date 1st Death by Alpha 3 Code')

fig.show()
import plotly.express as px

fig = px.bar(y=list(df['alpha_3_code'].value_counts().sort_index()), x=df['alpha_3_code'].value_counts().sort_index().index, title='Alpha 3 Code')

fig.show()
fig = px.scatter(df, x='deaths_per_million_55_days_after_first_death', y='date_first_death', title='Relation Price vs Fuel Type')

fig.show()
fig = px.scatter(df, x='deaths_per_million_100_days_after_first_death', y='date_first_death')

fig.show()
import plotly.graph_objects as go



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
aux = df.copy()

y = aux['date_first_death']

aux = aux.drop(['date_first_death'], axis=1)

X = aux
y = np.log1p(y) 
sns.distplot(y)
import warnings; warnings.simplefilter('ignore')

from sklearn.ensemble import StackingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import LinearSVR

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import StackingRegressor

from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y

)

estimators = [

    ('rf', SGDRegressor(random_state=42)),

    ('svr', LinearSVR(random_state=42))

]

clf = StackingRegressor(

    estimators=estimators, final_estimator=RandomForestRegressor()

)

model = clf.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

pred = model.predict(X_test)

np.sqrt(mean_squared_error(y_test, pred))
plot_predict(pred, y_test)
from sklearn.datasets import load_diabetes

from sklearn.linear_model import RidgeCV

from sklearn.svm import LinearSVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import StackingRegressor

estimators = [

    ('lr', RidgeCV()),

    ('svr', LinearSVR(random_state=42))

]

clf = StackingRegressor(

    estimators=estimators,

    final_estimator=RandomForestRegressor(n_estimators=10,

                                          random_state=42)

)



model = clf.fit(X_train, y_train)
pred = model.predict(X_test)

print('RMSE =',np.sqrt(mean_squared_error(y_test, pred)))
plot_predict(pred, y_test)
from sklearn.linear_model import Lasso

estimators = [

    ('rf', Lasso(alpha=0.1)),

    ('svr', LinearSVR(random_state=42))

]

clf = StackingRegressor(

    estimators=estimators, final_estimator=RandomForestRegressor()

)

model = clf.fit(X_train, y_train)

pred = model.predict(X_test)

print('RMSE =',np.sqrt(mean_squared_error(y_test, pred)))
plot_predict(pred, y_test)