import warnings
warnings.filterwarnings('ignore')

import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt

import seaborn as sns
color = sns.color_palette()

%matplotlib inline
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

from IPython.display import HTML, display
import tabulate

import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
%%time
train_df = load_df()
test_df = load_df("../input/test.csv")
print('size of training data : ', train_df.shape)
print('size of testing data  : ', test_df.shape)
train_df.head()
train_df.columns.values
test_df.head()
test_df.columns.values
total = train_df.isnull().sum().sort_values(ascending = False)
percent = (train_df.isnull().sum() / train_df.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_train_data.head(20)
temp1 = train_df['trafficSource.campaignCode'].value_counts()

trace1 = go.Bar(
    x = temp1.index,
    y = temp1 ,
)

data = [trace1]

layout = go.Layout(
    title = "Campaign code for training data",
    xaxis=dict(
        title='Campaign codes',
        domain=[0, 0.5]
    ),
    
    yaxis=dict(
        title='Count of Campaign codes '
        
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Campaign code')
train_df['trafficSource.campaignCode'].value_counts()
transactionRevenue = train_df['totals.transactionRevenue'].value_counts()
print(transactionRevenue.head())
len(transactionRevenue)
temp1 = train_df['trafficSource.adwordsClickInfo.page'].value_counts()
temp2 = test_df['trafficSource.adwordsClickInfo.page'].value_counts()


trace1 = go.Bar(
    x = temp1.index,
    y = temp1,
   
)
trace2 = go.Bar(
    x = temp2.index,
    y = temp2 
    
)

data = [trace1, trace2]

layout = go.Layout(
    title = "Page # where the ad was shown for training data",
    width = 900,
    xaxis=dict(
        title='Page #',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    
    yaxis=dict(
        title='# of instances',  
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
print(train_df['trafficSource.adwordsClickInfo.page'].value_counts())
print(test_df['trafficSource.adwordsClickInfo.page'].value_counts())
temp1 = train_df['trafficSource.adwordsClickInfo.adNetworkType'].value_counts()
temp2 = test_df['trafficSource.adwordsClickInfo.adNetworkType'].value_counts()


trace1 = go.Bar(
    x = temp1.index,
    y = temp1,
    name = 'train'
   
)
trace2 = go.Bar(
    x = temp2.index,
    y = temp2,
    name = 'test'
    
)

data = [trace1, trace2]

layout = go.Layout(
    title = "Page # where the ad was shown for training data",
    width = 900,
    xaxis=dict(
        title='Page #',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    
    yaxis=dict(
        title='# of instances',  
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
print(train_df['trafficSource.adwordsClickInfo.adNetworkType'].value_counts())
print(test_df['trafficSource.adwordsClickInfo.adNetworkType'].value_counts())
temp1 = train_df['trafficSource.adwordsClickInfo.slot'].value_counts()
temp2 = test_df['trafficSource.adwordsClickInfo.slot'].value_counts()


trace1 = go.Bar(
    x = temp1.index,
    y = temp1,
    name = 'train'
   
)
trace2 = go.Bar(
    x = temp2.index,
    y = temp2,
    name = 'test'
    
)

data = [trace1, trace2]

layout = go.Layout(
    title = "Page # where the ad was shown for training data",
    width = 900,
    xaxis=dict(
        title='Page #',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    
    yaxis=dict(
        title='# of instances',  
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
print(train_df['trafficSource.adwordsClickInfo.slot'].value_counts())
print(test_df['trafficSource.adwordsClickInfo.slot'].value_counts())

print(train_df['trafficSource.adwordsClickInfo.isVideoAd'].value_counts())
print(test_df['trafficSource.adwordsClickInfo.isVideoAd'].value_counts())
print(train_df['trafficSource.isTrueDirect'].value_counts())
print(test_df['trafficSource.isTrueDirect'].value_counts())

#print(train_df['trafficSource.referralPath'].value_counts())
#print(test_df['trafficSource.referralPath'].value_counts())
# print(train_df['trafficSource.keyword'].value_counts())
# print(test_df['trafficSource.keyword'].value_counts())
print(train_df['totals.bounces'].value_counts())
print(test_df['totals.bounces'].value_counts())
print(train_df['totals.newVisits'].value_counts())
print(test_df['totals.newVisits'].value_counts())