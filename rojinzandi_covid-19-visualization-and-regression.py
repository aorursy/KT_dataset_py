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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import time

from datetime import datetime

from scipy import integrate, optimize

import warnings

warnings.filterwarnings('ignore')



# ML libraries

import lightgbm as lgb

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import linear_model

from sklearn.preprocessing import scale

import sklearn.linear_model as skl_lm

from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm

import statsmodels.formula.api as smf



#Libraries to import



import datetime as dt

import requests

import sys

from itertools import chain

import plotly_express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import OrdinalEncoder

from sklearn import metrics

import xgboost as xgb

from xgboost import XGBRegressor

from xgboost import plot_importance, plot_tree

from sklearn.model_selection import GridSearchCV
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")



display(train.head())

display(train.describe())

train.info()

train.isnull().sum()

test.isnull().sum()



print("Number of Country_Region: ", train['Country_Region'].nunique())

print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")

print("Countries with Province/State informed: ", train.loc[train['Province_State']!='None']['Country_Region'].unique())
test.isnull().sum()
ID=train['Id']

FID=test['ForecastId']
train_date_min = train['Date'].min()

train_date_max = train['Date'].max()

print('Minimum date from training set: {}'.format(train_date_min))

print('Maximum date from training set: {}'.format(train_date_max))
test_date_min = test['Date'].min()

test_date_max = test['Date'].max()

print('Minimum date from test set: {}'.format(test_date_min))

print('Maximum date from test set: {}'.format(test_date_max))
sns.pairplot(train)
fig = px.pie(train, values='TargetValue', names='Target')

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
fig = px.pie(train, values='TargetValue', names='Country_Region')

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
fig = px.pie(train, values='Population', names='Country_Region')

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
corr_matrix = train.corr()     #computing correlation between features and output

print(corr_matrix)
#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = train.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
train=train.drop(columns=['County','Province_State','Id'])

test=test.drop(columns=['County','Province_State','ForecastId'])
da= pd.to_datetime(train['Date'], errors='coerce')

train['Date']= da.dt.strftime("%Y%m%d").astype(int)

da= pd.to_datetime(test['Date'], errors='coerce')

test['Date']= da.dt.strftime("%Y%m%d").astype(int)
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()

X = train.iloc[:,0].values

train.iloc[:,0] = l.fit_transform(X.astype(str))



X = train.iloc[:,4].values

train.iloc[:,4] = l.fit_transform(X)
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()

X = test.iloc[:,0].values

test.iloc[:,0] = l.fit_transform(X.astype(str))



X = test.iloc[:,4].values

test.iloc[:,4] = l.fit_transform(X)
y_train=train['TargetValue']

x_train=train.drop(['TargetValue'],axis=1)



from sklearn.model_selection import train_test_split 



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(x_train,y_train)



print(lin_reg.intercept_)

print(lin_reg.coef_)
acc1=lin_reg.score(x_test,y_test)

acc1
from sklearn.preprocessing import PolynomialFeatures

poly_reg2=PolynomialFeatures(degree=2)

x_poly=poly_reg2.fit_transform(x_train)

lin_reg_2=LinearRegression()

lin_reg_2.fit(x_poly,y_train)



print("Coefficients of polynimial(degree2) are", lin_reg_2.coef_)
#comparing estimators

from sklearn.ensemble import RandomForestRegressor 

model = RandomForestRegressor(n_jobs=-1)

estimators = np.arange(10, 200, 10)

scores = []

for n in estimators:

    model.set_params(n_estimators=n)

    model.fit(x_train, y_train)

    scores.append(model.score(x_test, y_test))

plt.title("Effect of n_estimators")

plt.xlabel("n_estimator")

plt.ylabel("score")

plt.plot(estimators, scores)
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

p = Pipeline([('scaler2' , StandardScaler()),

                        ('RandomForestRegressor: ', RandomForestRegressor())])

p.fit(x_train , y_train)

prediction = p.predict(x_test)
acc2=p.score(x_test,y_test)

acc2
predict=p.predict(test)
output=pd.DataFrame({'id':FID,'TargetValue':predict})

output
a=output.groupby(['id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']

a
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()