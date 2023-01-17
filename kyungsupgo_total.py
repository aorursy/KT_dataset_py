import datetime

from datetime import datetime

import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats



plt.style.use('seaborn')





import missingno as msno

import warnings

warnings.filterwarnings('ignore')





%matplotlib inline
df_train= pd.read_csv('../input/train.csv', parse_dates=['datetime'])

df_test = pd.read_csv('../input/test.csv', parse_dates=['datetime'])

df_train.head(10)
df_train.info()
df_train.describe()
# Train 과 Test data의 Null값 확인
msno.matrix(df= df_train.iloc[: , :],figsize=(8,8), color = (0.8,0.5,0.2))
msno.matrix(df= df_test.iloc[: , :],figsize=(8,8), color = (0.8,0.5,0.2))
df_train['year']=df_train['datetime'].dt.year

df_train['month']=df_train['datetime'].dt.month

df_train['hour']=df_train['datetime'].dt.hour

df_train['weekday']=df_train['datetime'].dt.dayofweek
df_test['year']=df_test['datetime'].dt.year

df_test['month']=df_test['datetime'].dt.month

df_test['hour']=df_test['datetime'].dt.hour

df_test['weekday']=df_test['datetime'].dt.dayofweek
df_train.head(10)
#df_train.drop(['datetime'], axis=1, inplace=True)
fig, axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(18, 15)

sns.boxplot(data=df_train,y="count",orient="v",ax=axes[0][0])

sns.boxplot(data=df_train,y="count",x="season",orient="v",ax=axes[0][1])

sns.boxplot(data=df_train,y="count",x="hour",orient="v",ax=axes[1][0])

sns.boxplot(data=df_train,y="count",x="workingday",orient="v",ax=axes[1][1])



axes[0][0].set(ylabel='Count',title="Box Plot On Count")

axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")

axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")

axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")
from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



import plotly.graph_objs as go



import time

import random



def generate_color():

    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
### 유니크 갯수 계산

train_unique = []

columns = ['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered','count']



for i in columns:

    train_unique.append(len(df_train[i].unique()))

unique_train = pd.DataFrame()

unique_train['Columns'] = columns

unique_train['Unique_value'] = train_unique



data = [

    go.Bar(

        x = unique_train['Columns'],

        y = unique_train['Unique_value'],

        name = 'Unique value in features',

        textfont=dict(size=20),

        marker=dict(

        line=dict(

            color= generate_color(),

            #width= 2,

        ), opacity = 0.45

    )

    ),

    ]

layout= go.Layout(

        title= "Unique Value By Column",

        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),

        yaxis= dict(title='Value Count', ticklen=5, gridwidth=2),

        showlegend=True

    )

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='skin')
df_train['atemp'].unique()
df_train[['season','count']].groupby(['season'], as_index=True).mean().plot.bar()
df_train[['season','count']].groupby(['season'], as_index=True).mean().plot()
data = pd.concat([df_train['count'], df_train['season']], axis=1)

f, ax = plt.subplots(figsize=(15,15))

fig = sns.boxplot(x='season', y="count", data=data)
data = pd.concat([df_train['count'], df_train['season']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='season', y="count", data=data)
df_train.loc[df_train['count'] > 800]
df_train[['weather','count']].groupby(['weather'], as_index=True).mean().plot.bar()
data = pd.concat([df_train['count'], df_train['weather']], axis=1)

f, ax = plt.subplots(figsize=(15,15))

fig = sns.boxplot(x='weather', y="count", data=data)
data = pd.concat([df_train['count'], df_train['weather']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.regplot(x='weather', y="count", data=data)
df_train.loc[df_train['weather'] == 4]
fig, axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(18, 15)

sns.regplot(x='temp', y='count', data=df_train,ax=axes[0][0])

sns.regplot(x='atemp', y='count', data=df_train,ax=axes[0][1])

sns.regplot(x='windspeed', y='count', data=df_train,ax=axes[1][0])

sns.regplot(x='humidity', y='count', data=df_train,ax=axes[1][1])



axes[0][0].set(xlabel='temp',ylabel='Count',title='temp ')

axes[0][1].set(xlabel='atemp', ylabel='Count',title='atemp')

axes[1][0].set(xlabel='windspeed', ylabel='Count',title='windspeed')

axes[1][1].set(xlabel='humidity', ylabel='Count',title='humidity')
fig, axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(18, 15)





sns.distplot(df_train['temp'],ax=axes[0][0] )

sns.distplot(df_train['atemp'],ax=axes[0][1])

sns.distplot(df_train['windspeed'],ax=axes[1][0])

sns.distplot(df_train['humidity'],ax=axes[1][1])





axes[0][0].set(xlabel='temp',ylabel='Count',title=['Temp',("Skewness: %f" % df_train['temp'].skew()),("Kurtosis: %f" % df_train['temp'].kurt())])





axes[0][1].set(xlabel='atemp', ylabel='Count',title=['Atemp',("Skewness: %f" % df_train['temp'].skew()),("Kurtosis: %f" % df_train['atemp'].kurt())])

axes[1][0].set(xlabel='windspeed', ylabel='Count',title=['Windspeed',("Skewness: %f" % df_train['temp'].skew()),("Kurtosis: %f" % df_train['windspeed'].kurt())])

axes[1][1].set(xlabel='humidity', ylabel='Count',title=['Humidity',("Skewness: %f" % df_train['temp'].skew()),("Kurtosis: %f" % df_train['humidity'].kurt())])
f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['count'])
df_train.loc[df_train['count'] < 10]
fig = plt.figure(figsize = (15,10))



fig.add_subplot(1,2,1)

res = stats.probplot(df_train['count'], plot=plt)



fig.add_subplot(1,2,2)

res = stats.probplot(np.log1p(df_train['count']), plot=plt)
df_train['count'] = np.log1p(df_train['count'])

#histogram

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(df_train['count'])
# correlation이 높은 상위 10개의 heatmap

# continuous + sequential variables --> spearman

# abs는 반비례관계도 고려하기 위함

# https://www.kaggle.com/junoindatascience/let-s-eda-it 준호님이 수정해 준 코드로 사용하였습니다. 

import scipy as sp



cor_abs = abs(df_train.corr(method='spearman')) 

cor_cols = cor_abs.nlargest(n=10, columns='count').index # count과 correlation이 높은 column 10개 뽑기(내림차순)

# spearman coefficient matrix

cor = np.array(sp.stats.spearmanr(df_train[cor_cols].values))[0] # 10 x 10

print(cor_cols.values)

plt.figure(figsize=(10,10))

sns.set(font_scale=1.25)

sns.heatmap(cor, fmt='.2f', annot=True, square=True , annot_kws={'size' : 8} ,xticklabels=cor_cols.values, yticklabels=cor_cols.values)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score



from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
#df_test['date']  = df_test.datetime.apply(lambda x: x.split()[0])

#df_test['hour'] = df_test.datetime.apply(lambda x: x.split()[1].split(':')[0])

#df_test['weekday'] = df_test.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())

#df_test['month'] = df_test.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)

#df_test['year']=[d.split('-')[0] for d in df_test.date]

#df_test['day']=[d.split('-')[2] for d in df_test.date]
df_train_care=df_train[['casual','registered']]

del df_train['casual']

del df_train['registered']
df_train.head(10)
target_label = df_train['count']



del df_train['count']
train_len = len(df_train)

df_train = pd.concat((df_train, df_test), axis=0)
msno.matrix(df= df_train.iloc[: , :],figsize=(8,8), color = (0.8,0.5,0.2))
df_test_datetime=df_train['datetime'][train_len:]
## 여기서부터 컬럼 형태바꾸고 필요없는 컬럼 drop

df_train.dtypes
from sklearn.preprocessing import  LabelEncoder

encoder =LabelEncoder()
for i in ['season','holiday','month','weather','weekday','workingday']:

    encoder.fit(df_train[i])

    df_train[i]=encoder.transform(df_train[i])
df_train.dtypes
#불필요한 컬럼 제거 

df_train=df_train.drop(['year','datetime'],axis=1)
df_test = df_train.iloc[train_len:, :]

df_train = df_train.iloc[:train_len, :]
X_train = df_train.values

# target_label은 들어 가있으니까 그대로 유지

X_test = df_test.values

# array형태로 넣어주려고 values 사용
lgbm=LGBMRegressor()

xgb=XGBRegressor()
xgb.fit(df_train,target_label)
lgbm.fit(df_train,target_label)
lgbm_pred = np.expm1(lgbm.predict(df_test))

xgb_pred=np.expm1(xgb.predict(df_test))
preds=0.5*xgb_pred + 0.5*lgbm_pred
df_test.head(10)
type(df_test_datetime)
sub = pd.DataFrame(data={'datetime':df_test_datetime,'count':preds})
sub.to_csv('submission.csv',index=False)