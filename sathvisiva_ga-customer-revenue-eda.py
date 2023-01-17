# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import bq_helper
from pandas.io.json import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff
import numpy as np
init_notebook_mode(connected = True)
import datetime
import missingno as msno


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

nan_list = ["not available in demo dataset",
            "unknown.unknown",
            "(not provided)",
            "(not set)", 
            "Not Socially Engaged"] # this last one is borderline
nan_dict = {nl:np.nan for nl in nan_list}

def load_data(file):
    path = '../input/' + file
    df = pd.read_csv(path, converters = {column: json.loads for column in json_cols},
                     dtype={'fullVisitorId': str, 'date': str}, 
            parse_dates=['date'])
    
    for column in json_cols:
        dataframe = json_normalize(df[column])
        dataframe.columns = [f"{column}_{subcolumn}" for subcolumn in dataframe.columns]
        df = df.drop(column, axis = 1).merge(dataframe, right_index = True, left_index = True)
        df.replace(nan_dict, inplace=True)
    return df

train = load_data("train.csv")
#test = load_data('test.csv')
train.head()
def plotlybar(labels1 = None, values1 = None, name1 = None,labels2 = None, values2 = None, name2 =None, markercolor1 = '#9ECAE1', markercolor2 = '#ff99ff', title= None, mode = 'group', orientation = 'v'):
    trace1 = go.Bar(
    x = labels1,
    y = values1,
    orientation = orientation, 
    text = values1,
    name = name1,
    textposition = 'auto',
    marker=dict(
        color=markercolor1#'rgb(58,200,225)',
        )
    )    
    
    if labels2 is not None:
        trace2 = go.Bar(
            x = labels2,
            y = values2,
            name = name2,
            text = values2,
            textposition = 'auto',
            marker=dict(
                color=markercolor2#'rgb(58,200,225)',
            )
        )
        data = [trace1, trace2]
    
    else:
        data = [trace1]
    layout = go.Layout(title = title,xaxis=dict(tickangle=-25),
    barmode=mode)
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)
    
def scatter_plot(x,y, title):
    data = [go.Scatter(
    x = x,
    y = y,
    mode = 'lines+markers',
    name = 'lines+markers'
    )]
    layout = go.Layout(title = title,xaxis=dict(tickangle=-25))
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)
total = train.isnull().sum().sort_values(ascending = True)
percent = (train.isnull().sum() / train.isnull().count() * 100).sort_values(ascending = True)
missing_data = pd.concat([total, percent], axis = 1, keys = ['Total','Percent'])
missing_data
train.describe()
cols_to_drop = [column for column in train.columns if train[column].nunique()==1]
train = train.drop(cols_to_drop , axis=1)
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train['day'] = train['date'].dt.weekday
train['totals_transactionRevenue'] = pd.to_numeric(train['totals_transactionRevenue'])
train['totals_transactionRevenue'] = train['totals_transactionRevenue'].fillna(0)
train['totals_hits'] = pd.to_numeric(train['totals_hits']).fillna(0)
train['totals_pageviews'] = pd.to_numeric(train['totals_pageviews']).fillna(0)
plt.figure(figsize=(12,6))
sns.distplot(train[train['totals_transactionRevenue'] > 0]['totals_transactionRevenue'],color="darkgreen",bins=50)
plt.xlabel("total transaction revenue");
plt.title("Distribution of total transaction revenue (non-zeros)");
plt.figure(figsize=(12,6))
sns.distplot(np.log1p(train[train['totals_transactionRevenue'] > 0]['totals_transactionRevenue']),color="darkgreen",bins=50)
plt.xlabel("total transaction revenue");
plt.title("Lograthemic Distribution of total transaction revenue (non-zeros)");
train_ = train.loc[train['totals_transactionRevenue'] > 0]
fig, ax1 = plt.subplots(figsize=(16, 8))
plt.title("Revenue and Non Revenue transactions");
train.groupby(['date'])['totals_transactionRevenue'].count().plot(color='#E7638B')
ax1.set_ylabel('Transaction count', color='#E7638B')
plt.legend(['Non-Revenue users'])
ax2 = ax1.twinx()
train_.groupby(['date'])['totals_transactionRevenue'].count().plot(color='#7debca')
ax2.set_ylabel('Transaction count', color='#7debca')
plt.legend(['Revenue users'], loc=(0.875, 0.9))
plt.grid(False)
group_by_date = train.groupby('date')['fullVisitorId'].size()
scatter_plot(group_by_date.index, group_by_date.values, 'Visitors by date')

visit_by_month = train['month'].value_counts()
visits_by_day = train['day'].value_counts()
plotlybar(visit_by_month.index, visit_by_month.values, 'Month', markercolor1 = '#E7638B' , title = 'Visits By Month')
plotlybar(visits_by_day.index[:10], visits_by_day.values[:10], 'Day', markercolor1 = '#7dd5eb' , title = 'Visits By Day')
train['time_of_visit'] = train['visitStartTime'].apply(lambda x : x % (24 * 3600) // 3600)
time_of_visit = train['time_of_visit'].value_counts()
#time_of_visit
plotlybar(time_of_visit.index, time_of_visit.values, 'Time', markercolor1 = '#7debca' , title = 'Time of Visit')
Channels = train['channelGrouping'].value_counts()
plotlybar(Channels.index, Channels.values, 'Channel Grouping', markercolor1 = '#E7638B' , title = 'Count of Channel Grouping')
monthy_chnl_rev = train.groupby(['channelGrouping','month'])['totals_transactionRevenue'].mean().reset_index()

fig = {
    'data': [
        {
            'x' : monthy_chnl_rev[monthy_chnl_rev['channelGrouping']==channel]['month'],
            'y': monthy_chnl_rev[monthy_chnl_rev['channelGrouping']==channel]['totals_transactionRevenue'],
            'name': channel
        } for channel in ['Organic Search', 'Social', 'Direct', 'Referral', 'Paid Search', 'Affiliates', 'Display', '(Other)']
    ],
    'layout': {
        'title' : 'Month wise mean revenue from each Channel Grouping',
        'xaxis': {'title': 'Month'},
        'yaxis': {'title': "Mean Revenue"}
    }
}
iplot(fig)
Browsers = train['device_browser'].value_counts()
plotlybar(Browsers.index[:10], Browsers.values[:10], 'Browsers', markercolor1 = '#7debca' , title = 'Types of browsers used')
#device_deviceCategory
Device_category = train['device_deviceCategory'].value_counts()
plotlybar(Device_category.index, Device_category.values, 'Device category', markercolor1 = '#7dd5eb' , title = 'Category of Devices')
monthy_dev_rev = train.groupby(['device_deviceCategory','month'])['totals_transactionRevenue'].mean().reset_index()

fig = {
    'data': [
        {
            'x' : monthy_dev_rev[monthy_dev_rev['device_deviceCategory']==device]['month'],
            'y': monthy_dev_rev[monthy_dev_rev['device_deviceCategory']==device]['totals_transactionRevenue'],
            'name': device
        } for device in ['desktop', 'mobile', 'tablet']
    ],
    'layout': {
        'title' : 'Month wise mean revenue from each device category',
        'xaxis': {'title': 'Month'},
        'yaxis': {'title': "Mean Revenue"}
    }
}
iplot(fig)
os = train['device_operatingSystem'].value_counts()
plotlybar(os.index, os.values, 'os', markercolor1 = '#7debca' , title = 'Different OS')
#device_deviceCategory
geo_continent = train['geoNetwork_continent'].value_counts()
plotlybar(geo_continent.index, geo_continent.values, 'continent', markercolor1 = '#7debca' , title = 'Country wise count')
#device_deviceCategory
geo_country = train['geoNetwork_country'].value_counts()
plotlybar(geo_country.index[:10], geo_country.values[:10], 'country', markercolor1 = '#E7638B' , title = 'Top 10 countries')
network_country = train["geoNetwork_country"].value_counts()
colorscale = [[0, '#14f962'], [0.005, '#15ed11'], 
              [0.01, '#5fe10f'], [0.02, '#a2d50d'], 
              [0.04, '#c9b50b'], [0.05, '#bd6909'], 
              [0.10, '#b12407'], [0.25, '#a50624'], [1.0, '#990557']]

data = [
    dict(
    type = 'choropleth',
        autocolorscale = False,
        colorscale = colorscale,
        showscale = True,
        locations = network_country.index,
        z = network_country.values,
        locationmode = "country names",
        text = network_country.values,
        marker = dict(
            line = dict(color = '#fff' , width = 2)
        )
        
    )
]

layout = dict(
    height = 700,
    title = "Geo Network countries",
    geo =dict(
        showframe = True,
        showocean = True,
        oceancolor = '#0077be',
        projection = dict(
            type = 'orthographic',
            rotation = dict(
                lon = 60,
                lat = 10
            )
        ),
        lonaxis = dict(
            showgrid = False,
            gridcolor = 'rgb(102, 102, 102)'
        ),
        
        lataxis = dict(
            showgrid = False,
            gridcolor = 'rgb(102,102,102)'
        )
    )
)

fig = dict(data= data, layout = layout)
iplot(fig)

monthy_geo_rev = train.groupby(['geoNetwork_continent','month'])['totals_transactionRevenue'].mean().reset_index()

fig = {
    'data': [
        {
            'x' : monthy_geo_rev[monthy_geo_rev['geoNetwork_continent']==geo]['month'],
            'y': monthy_geo_rev[monthy_geo_rev['geoNetwork_continent']==geo]['totals_transactionRevenue'],
            'name': geo
        } for geo in ['Americas', 'Asia', 'Europe','Oceania','Africa']
    ],
    'layout': {
        'title' : 'Month wise mean revenue from each geography region',
        'xaxis': {'title': 'Month'},
        'yaxis': {'title': "Mean Revenue"}
    }
}
iplot(fig)
plt.figure(figsize = (18,8))
plt.figure(1)
sns.set(style="darkgrid")
plt.subplot(221)
sns.countplot(x="trafficSource_adwordsClickInfo.adNetworkType", data=train ,color="#7dd5eb")
plt.subplot(222)
sns.countplot(x="trafficSource_adwordsClickInfo.slot", data=train ,color="#7dd5eb" )
plt.subplot(223)
sns.countplot(x="trafficSource_campaign", data=train ,color="#7dd5eb")
plt.xticks(rotation = 90)
plt.subplot(224)
sns.countplot(x="trafficSource_medium", data=train ,color="#7dd5eb" )
plt.xticks(rotation = 90)
plt.show()