!pip install pandas-profiling
import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import lightgbm as lgb

import xgboost as xgb

import time

import datetime

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit

from sklearn import metrics

from sklearn import linear_model

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



import eli5

import shap

from IPython.display import HTML

import json

# import altair as alt



import networkx as nx

import matplotlib.pyplot as plt

%matplotlib inline



# alt.renderers.enable('notebook')

import pandas_profiling

import plotly.graph_objs as go

# import plotly.plotly as py

import plotly.offline as pyo

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly_express as px

init_notebook_mode(connected=True)

from matplotlib import cm

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn.decomposition import PCA

from sklearn.cluster import KMeans



import plotly_express as px
folder_path = '../input/'

train_identity = pd.read_csv(f'{folder_path}train_identity.csv')

train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')

# test_identity = pd.read_csv(f'{folder_path}test_identity.csv')

# test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv')

sub = pd.read_csv(f'{folder_path}sample_submission.csv')

# let's combine the data and work with the whole dataset

# I will save this for later

# train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

# test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
print(f'Train Transaction dataset has {train_transaction.shape[0]} rows and {train_transaction.shape[1]} columns.')
# print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')

# print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')
train_transaction = train_transaction.sample(n=100000)

train_transaction.head()
trx_colnames = train_transaction.columns

for i in range(len(trx_colnames)):

    print(i, ': ',trx_colnames[i])
trx_colnames_main = trx_colnames[1:17]

trx_colnames_CDM = trx_colnames[17:55]

trx_colnames_V = trx_colnames[55:]
# profiler1 = train_transaction[trx_colnames_main].profile_report()

# profiler1
# profiler2 = train_transaction[trx_colnames_CDM].profile_report()

# profiler2
import sys



# These are the usual ipython objects, including this one you are creating

ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']



# Get a sorted list of the objects and their sizes

sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
trx_colnames_C = [c for c in trx_colnames if c.startswith("C") ]

trx_colnames_C
trxC = train_transaction[trx_colnames_C]

trxClog = np.log1p(trxC)

fig, ax = plt.subplots(15,3,figsize=(10,20))

for i,c in enumerate(trxC.columns):

    trxC[c].hist(ax=ax[i,0])

    trxClog[c].hist(ax=ax[i,1])

    trxClog[c].hist(ax=ax[i,2],cumulative=True,density=True)

plt.tight_layout()

plt.suptitle('Distribution of C variables - C, log(C), cumulative log(C)')
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
X = trxClog

pca = PCA(n_components=len(trx_colnames_C)).fit(X)

#Plotting the Cumulative Summation of the Explained Variance

expvar=np.cumsum(pca.explained_variance_ratio_)

data = [go.Scatter(y=expvar)]

layout = {'title': 'Review PCA Explained Variance to determine number of components'}

iplot({'data':data,'layout':layout})
pca = PCA(n_components=4)

XPCA = pca.fit_transform(X)

Nc = range(1,10)

kmeans = [KMeans(i) for i in Nc]

score = [kmeans[i].fit(XPCA).score(XPCA) for i in range(len(kmeans))]

data = [go.Scatter(y=score,x=list(Nc))]

layout = {'title':'Review Elbow Curve to determine number of clusters for KMeans'}

iplot({'data':data,'layout':layout})
from yellowbrick.features.pca import PCADecomposition

visualizer = PCADecomposition(scale=True)

visualizer.fit_transform(trxClog)

visualizer.poof()
trx_colnames_V = [c for c in trx_colnames if c.startswith("V") ]

trx_colnames_V
trxV = train_transaction[trx_colnames_V]

trxVlog = np.log1p(trxV)

fig, ax = plt.subplots(15,3,figsize=(10,20))

for i,c in enumerate(trxV.columns[:15]):

    trxV[c].hist(ax=ax[i,0],bins=50)

    trxVlog[c].hist(ax=ax[i,1],bins=50)

    trxVlog[c].hist(ax=ax[i,2],bins=50,cumulative=True,density=True,histtype='step')

plt.tight_layout()

plt.suptitle('Distribution of V variables - V, log(V), cumulative log(V)')
X = trxV.fillna(0)

pca = PCA(n_components=10).fit(X)

#Plotting the Cumulative Summation of the Explained Variance

expvar=np.cumsum(pca.explained_variance_ratio_)

data = [go.Scatter(y=expvar)]

layout = {'title': 'Review PCA Explained Variance to determine number of components'}

iplot({'data':data,'layout':layout})
from yellowbrick.features.pca import PCADecomposition

visualizer = PCADecomposition(scale=True)

visualizer.fit_transform(trxV.fillna(0))

visualizer.poof()
def make_day_feature(df, offset=0, tname='TransactionDT'):

    """

    Creates a day of the week feature, encoded as 0-6. 

    

    Parameters:

    -----------

    df : pd.DataFrame

        df to manipulate.

    offset : float (default=0)

        offset (in days) to shift the start/end of a day.

    tname : str

        Name of the time column in df.

    """

    # found a good offset is 0.58

    days = df[tname] / (3600*24)        

    encoded_days = np.floor(days-1+offset) % 7

    return encoded_days



def make_hour_feature(df, tname='TransactionDT'):

    """

    Creates an hour of the day feature, encoded as 0-23. 

    

    Parameters:

    -----------

    df : pd.DataFrame

        df to manipulate.

    tname : str

        Name of the time column in df.

    """

    hours = df[tname] / (3600)        

    encoded_hours = np.floor(hours) % 24

    return encoded_hours



START_DATE = '2017-12-01'

startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
train_transaction['TransactionDateTime'] = train_transaction['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

train_transaction['TransactionDate'] = [x.date() for x in train_transaction['TransactionDateTime']]

# train_transaction['TransactionDateHour'] = train_transaction['TransactionDateTime'].date()

train_transaction['TransactionHour'] = train_transaction.TransactionDT // 3600

train_transaction['TransactionHourOfDay'] = train_transaction['TransactionHour'] % 24

train_transaction['TransactionDay'] = train_transaction.TransactionDT // (3600 * 24)
train_transaction.head(10)
agg_dict = {}

for col in trx_colnames_C:

    agg_dict[col] = ['mean','sum']

print(agg_dict)
train_trx_hour = train_transaction.groupby(['TransactionHour']).agg(agg_dict).reset_index()

train_trx_hour.head(5)
train_trx_hour.columns.values
train_trx_hour.columns = ['_'.join(col).strip() for col in train_trx_hour.columns.values]

train_trx_hour.head()
train_trx_hour.columns = ['TrxHourAgg_' + col for col in train_trx_hour.columns.values]

train_trx_hour.head()
train_trx_hour.DShour = pd.to_datetime(train_trx_hour.TransactionHour)

train_trx_hour.head()
train_trx_day = train_transaction.groupby(['TransactionDay'])["isFraud","TransactionAmt"].mean().reset_index()

train_trx_day.head()
fig = px.line(train_trx_hour.iloc[:1000,:],x="TransactionHour",y="isFraud",title='Average Fraud Rate by Hour')

fig.show()
train_trx_date = train_transaction.groupby(['TransactionDate']).agg({'isFraud':['mean','sum'],'TransactionAmt':['count','mean','sum']}).reset_index()

train_trx_date.head()
train_trx_date.columns = train_trx_date.columns.get_level_values(0) + '_' + train_trx_date.columns.get_level_values(1)

train_trx_date.head()
fig = px.line(train_trx_date,x="TransactionDate_",y="isFraud_sum",title="Total Frauds by Date")

fig.show()
px.scatter(train_trx_date,x="TransactionAmt_count",y="isFraud_mean",trendline='lowess',title="Relationship between Transaction Amount and Fraud Rate")
px.scatter(train_trx_date,x="TransactionAmt_count",y="isFraud_sum",trendline='lowess',title='Relationship between Transaction Amount and Number of Frauds (abs)')
train_trx_date["DayOfWeek"] = [x.weekday() for x in train_trx_date.TransactionDate_]

train_trx_date["DayOfWeek"] = train_trx_date["DayOfWeek"].apply(str)

train_trx_date.head()
px.scatter(train_trx_date,x="TransactionAmt_count",y="isFraud_mean",color='DayOfWeek',marginal_y='box',trendline='ols',title="Fraud rate vs. Transaction Amount")
px.scatter(train_trx_date,x="TransactionAmt_count",y="isFraud_sum",color='DayOfWeek',marginal_y='box',trendline='ols',title="Fraud Count vs. Transaction Amount")
train_trx_day['TransactionDay'].dtype

datetime.date.fromordinal(2)
from fbprophet import Prophet

df = train_trx_date[['TransactionDate_','isFraud_sum']]

df.columns = ['ds','y']

prophet = Prophet()

prophet.fit(df)

forecast = prophet.predict(df)
forecast.head()
fig,ax = plt.subplots(1,3,figsize=(20,5))

forecast.weekly[:7].plot(ax=ax[0],ylim=(-2,40))

ax[0].set_title("weekly component")

forecast.trend.plot(ax=ax[1],ylim=(-2,40))

ax[1].set_title("trend component")

# ax[1].xticks(forecast.ds)

ax[2].plot(forecast.yhat)

ax[2].plot(df.y)

ax[2].set_ylim(-2,40)

ax[2].set_title("comparing fitted vs. actual")

plt.suptitle('Avg Fraud Count based on: Day Of Week, Long-term Trend, Seasonality vs. Actual')
df = train_trx_date[['TransactionDate_','isFraud_mean']]

df.columns = ['ds','y']

prophet = Prophet()

prophet.fit(df)

forecast = prophet.predict(df)



fig,ax = plt.subplots(1,3,figsize=(20,5))

forecast.weekly[:7].plot(ax=ax[0],ylim=(-0.01,0.09))

ax[0].set_title("weekly component")

forecast.trend.plot(ax=ax[1],ylim=(-0.01,0.09))

ax[1].set_title("trend component")

ax[2].plot(forecast.yhat)

ax[2].plot(df.y)

ax[2].set_ylim(-0.01,0.09)

ax[2].set_title("comparing fitted vs. actual")

plt.suptitle('Avg Fraud Rate based on: Day Of Week, Long-term Trend, Seasonality vs. Actual')