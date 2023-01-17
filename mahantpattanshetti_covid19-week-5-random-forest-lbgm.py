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

from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error

from pandas import DataFrame





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


test = pd.read_csv("../input/covid19-global-forecasting-week-5/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-5/train.csv")



display(train.head())

display(train.describe())

train.info()



print("Number of Country_Region: ", train['Country_Region'].nunique())

print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")

print("Countries with Province/State informed: ", train.loc[train['Province_State']!='None']['Country_Region'].unique())
train_date_min = train['Date'].min()

train_date_max = train['Date'].max()

print('Minimum date from training set: {}'.format(train_date_min))

print('Maximum date from training set: {}'.format(train_date_max))
test_date_min = test['Date'].min()

test_date_max = test['Date'].max()

print('Minimum date from test set: {}'.format(test_date_min))

print('Maximum date from test set: {}'.format(test_date_max))
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
train["Province_State"] = train.Province_State.fillna(0)

test["Province_State"] = test.Province_State.fillna(0)

train.head()
total = len(train["Country_Region"])

print(total)
for i in range(0,total):

    if(train.Province_State[i] == 0):

        train.Province_State[i] = train.Country_Region[i]

train.head()

        

        
total_test = len(test["Province_State"])

for i in range(0,total_test):

    if(test.Province_State[i] == 0):

        test.Province_State[i] = test.Country_Region[i]

test.head()
train["t2"] = train.Target.factorize()[0]

test["t2"] = test.Target.factorize()[0]

train.head()
test.head()
label = preprocessing.LabelEncoder()

train.Country_Region = label.fit_transform(train.Country_Region)

train.Province_State = label.fit_transform(train.Province_State)



train.head()
test.Country_Region = label.fit_transform(test.Country_Region)

test.Province_State = label.fit_transform(test.Province_State)

test.head()

from sklearn.preprocessing import OrdinalEncoder



def create_features(df):

    df['day'] = df['Date'].dt.day

    df['month'] = df['Date'].dt.month

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['quarter'] = df['Date'].dt.quarter

    df['weekofyear'] = df['Date'].dt.weekofyear

    return df



def train_dev_split(df, days):

    #Last days data as dev set

    date = df['Date'].max() - dt.timedelta(days=days)

    return df[df['Date'] <= date], df[df['Date'] > date]



test_date_min = test['Date'].min()

test_date_max = test['Date'].max()



def avoid_data_leakage(df, date=test_date_min):

    return df[df['Date']<date]



def to_integer(dt_time):

    return 10000*dt_time.year + 100*dt_time.month + dt_time.day



train['Date']=pd.to_datetime(train['Date'])

test['Date']=pd.to_datetime(test['Date'])



test['Date']=test['Date'].dt.strftime("%m%d").astype(int)

train['Date']=train['Date'].dt.strftime("%m%d").astype(int)

train.head()
test.head()
x = train.drop(['Id','County','Target', 'TargetValue'],axis=1)

x_test = test.drop(['ForecastId','County','Target'],axis=1)

y = train["TargetValue"]

x.head ()
x_test.head()
y.head()
from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(x,y)

print(lin_reg.intercept_)

print(lin_reg.coef_)
sns.pairplot(train, x_vars=['Population','Weight','Date'], y_vars='TargetValue', size=7, aspect=0.7, kind='reg')
from sklearn.preprocessing import PolynomialFeatures

poly_reg2=PolynomialFeatures(degree=2)

X_poly=poly_reg2.fit_transform(x)

lin_reg_2=LinearRegression()

lin_reg_2.fit(X_poly,y)



print("Coefficients of polynimial(degree2) are", lin_reg_2.coef_)
poly_reg3=PolynomialFeatures(degree=3)

X_poly3=poly_reg3.fit_transform(x)

lin_reg_3=LinearRegression()

lin_reg_3.fit(X_poly3,y)

print("Coefficients of polynimial(degree3) are", lin_reg_3.coef_)

from sklearn.ensemble import RandomForestRegressor 

model = RandomForestRegressor(n_jobs=-1)

estimators = [10,50,100]

scores = []

for n in estimators:

    model.set_params(n_estimators=n)

    model.fit(x,y)

    scores.append(model.score(x,y))

plt.title("Effect of n_estimators")

plt.xlabel("n_estimator")

plt.ylabel("score")

plt.plot(estimators, scores)
main_model=RandomForestRegressor(n_estimators=100, n_jobs=-1)

main_model.fit(x,y)

y_pred = main_model.predict(x)

y_pred = np.round(y_pred)
y_pred
mae = mean_squared_error(y_pred,y)

print("The mean absolute error is =", mae, "Training Error")
lbg = LGBMRegressor(n_estimators = 5000, learning_rate = 1.1,  random_state = 42 , max_depth = 18)

lbg.fit(x,y)

yp = lbg.predict(x)

yp = np.round(yp)

m = mean_squared_error(yp,y)

print("The mean absolute error is =", m, "Training Error")

output = lbg.predict(x_test)

output = np.round(output)
df = pd.DataFrame()

f = test["ForecastId"]

f = f.astype(int)

df.insert(0,"Id",f,False)

df.insert(1,"TargetValue",output,False)

df.head()
q=df.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

w=df.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

e=df.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()



q.columns=['Id','q0.05']

w.columns=['Id','q0.5']

e.columns=['Id','q0.95']

q=pd.concat([q,w['q0.5'],e['q0.95']],1)

q['q0.05']=q['q0.05'].clip(0,10000)

q['q0.5']=q['q0.5'].clip(0,10000)

q['q0.95']=q['q0.95'].clip(0,10000)

q['Id'] =q['Id']
sub=pd.melt(q, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.info()
sub.head()