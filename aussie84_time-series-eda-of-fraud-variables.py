import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

pd.options.display.precision = 15



import time

import datetime

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



import json

# import altair as alt

import matplotlib.pyplot as plt

%matplotlib inline

# alt.renderers.enable('notebook')



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
folder_path = '../input/ieee-fraud-detection//'

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
train_transaction = train_transaction.sample(n=10000)

train_transaction.head()
print(f'Train Transaction dataset has {train_transaction.shape[0]} rows and {train_transaction.shape[1]} columns.')
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

train_transaction['TransactionHour'] = train_transaction.TransactionDT // 3600

train_transaction['TransactionHourOfDay'] = train_transaction['TransactionHour'] % 24

train_transaction['TransactionDay'] = train_transaction.TransactionDT // (3600 * 24)
train_transaction.head(10)
trx_colnames = train_transaction.columns

trx_colnames_core_num = ['isFraud', 'TransactionAmt', 'card1','card2', 'card3','card5']

trx_colnames_core_cat = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain']

trx_colnames_C = [c for c in trx_colnames if c.startswith("C") ]

trx_colnames_V = [c for c in trx_colnames if c.startswith("V") ]

trx_colnames_M = [c for c in trx_colnames if c.startswith("M") ]
agg_dict = {}

for col in trx_colnames_core_num:

    agg_dict[col] = ['mean','sum']

train_trx_hour = train_transaction.groupby(['TransactionHour']).agg(agg_dict).reset_index()

train_trx_hour.columns = ['_'.join(col).strip() for col in train_trx_hour.columns.values]

train_trx_hour.head()
import math

df = train_trx_hour

plotted_columns = train_trx_hour.columns[1:]

lencols = len(plotted_columns)

fig, axes = plt.subplots(math.ceil(lencols//3),3,figsize=(12,lencols))

for i, metrics in enumerate(plotted_columns):

    df.plot(x='TransactionHour_',y=metrics,title=metrics + " by Hour",ax=axes[i//3,i%3])

plt.tight_layout()

plt.suptitle("Core Metrics on Hourly Basis",y="1.05")

fig.show()
agg_dict = {}

for col in trx_colnames_core_num:

    agg_dict[col] = ['mean','sum']

train_trx_date = train_transaction.groupby(['TransactionDate']).agg(agg_dict).reset_index()

train_trx_date.columns = ['_'.join(col).strip() for col in train_trx_date.columns.values]

train_trx_date.head()
pd.plotting.register_matplotlib_converters()

import math

df = train_trx_date

plotted_columns = df.columns[1:]

lencols = len(plotted_columns)

fig, axes = plt.subplots(math.ceil(lencols//3),3,figsize=(16,lencols))

for i, metrics in enumerate(plotted_columns):

    df.plot(x='TransactionDate_',y=metrics,title=metrics + " by Date",ax=axes[i//3,i%3])

plt.tight_layout()

plt.suptitle("Core Metrics on Date Basis",y="1.05")

fig.show()
from fbprophet import Prophet

prophet = Prophet()



df = train_trx_date[['TransactionDate_','isFraud_sum']]

df.columns = ['ds','y']

prophet = Prophet()

prophet.fit(df)

forecast = prophet.predict(df)
from fbprophet import Prophet



def plot_forecast(df_input,metrics):

    prophet = Prophet()

    df = df_input[['TransactionDate_',metrics]]

    df.columns = ['ds','y']

    prophet.fit(df)

    forecast = prophet.predict(df)

    fig,ax = plt.subplots(1,3,figsize=(20,5),sharey=True)

    forecast.weekly[:7].plot(ax=ax[0])

    ax[0].set_title("weekly component")

    forecast.trend.plot(ax=ax[1])

    ax[1].set_title("trend component")

#     ax[1].xticks(forecast.ds)

    ax[2].plot(forecast.yhat)

    ax[2].plot(df.y)

    ax[2].set_title("comparing fitted vs. actual")

    plt.suptitle(metrics + ' based on: Day Of Week, Long-term Trend, Seasonality vs. Actual')
for metrics in plotted_columns:

    plot_forecast(train_trx_date,metrics)
agg_dict = {}

for col in trx_colnames_C:

    agg_dict[col] = ['mean','sum']

train_trx_date = train_transaction.groupby(['TransactionDate']).agg(agg_dict).reset_index()

train_trx_date.columns = ['_'.join(col).strip() for col in train_trx_date.columns.values]

train_trx_date.head()
import math

pd.plotting.register_matplotlib_converters()

df = train_trx_date

plotted_columns = df.columns[1:]

lencols = len(plotted_columns)

fig, axes = plt.subplots(math.ceil(lencols//3)+1,3,figsize=(16,lencols))

for i, metrics in enumerate(plotted_columns):

    df.plot(x='TransactionDate_',y=metrics,title=metrics + " by Date",ax=axes[i//3,i%3])

plt.tight_layout()

plt.suptitle("Core Metrics on Date Basis",y="1.05")

fig.show()
import datetime

t = datetime.date(2018,1,1)

train_trx_date2 = train_trx_date[train_trx_date['TransactionDate_']>t].reset_index(drop=True)

train_trx_date2.head()
for metrics in plotted_columns:

    plot_forecast(train_trx_date2,metrics)
def percentile(n):

    def percentile_(x):

        return np.percentile(x, n)

    percentile_.__name__ = 'percentile_%s' % n

    return percentile_



agg_dict = {}

for col in trx_colnames_core_num:

    agg_dict[col] = ['mean','sum']

train_trx_HOD = train_transaction.groupby(['TransactionHourOfDay']).agg(agg_dict).reset_index()

train_trx_HOD.columns = ['_'.join(col).strip() for col in train_trx_HOD.columns.values]

train_trx_HOD.head()



agg_dict = {}

for col in trx_colnames_core_num:

    agg_dict[col] = [percentile(25),'median',percentile(75)]

train_trx_HOD2 = train_transaction.groupby(['TransactionHourOfDay']).agg(agg_dict).reset_index()

train_trx_HOD2.columns = ['_'.join(col).strip() for col in train_trx_HOD2.columns.values]

train_trx_HOD2.head()
pd.plotting.register_matplotlib_converters()

import math

df = train_trx_HOD

plotted_columns = df.columns[1:]

lencols = len(plotted_columns)

fig, axes = plt.subplots(math.ceil(lencols//2),2,figsize=(16,lencols))

for i, metrics in enumerate(plotted_columns):

    df.plot(x='TransactionHourOfDay_',y=metrics,title=metrics + " by HourOfDay",ax=axes[i//2,i%2])

plt.tight_layout()

plt.suptitle("Core Metrics by HourOfDay",y="1.05")

fig.show()