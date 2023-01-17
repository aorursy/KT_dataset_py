import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, download_plotlyjs
import plotly.graph_objs as go
from plotly import tools
import matplotlib.pyplot as plt
init_notebook_mode(connected=True)
from plotly.tools import FigureFactory as ff
import random
from collections import Counter
import warnings
import json
import os
import datetime
from pandas.io.json import json_normalize
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
import pycountry
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# train_df = pd.read_csv('flatten_train.csv')
# test_df = pd.read_csv('flatten_test.csv')
# helper functions
def constant_cols(df):
    cols = []
    columns = df.columns.values
    for col in columns:
        if df[col].nunique(dropna = False) == 1:
            cols.append(col)
    return cols

def diff_cols(df1,df2):
    columns1 = df1.columns.values
    columns2 = df2.columns.values
    print(list(set(columns1) - set(columns2)))
    

def count_mean(col,color1,color2):
    col_count = train_df[col].value_counts()
    col_count_chart = go.Bar(x = col_count.head(10).index, y = col_count.head(10).values, name="Count",marker = dict(color=color1))

    col_mean_count = train_df[[col,'totals.transactionRevenue']][(train_df['totals.transactionRevenue'] >1)]
    col_mean_count = col_mean_count.groupby(col)['totals.transactionRevenue'].mean().sort_values(ascending=False)
    col_mean_count_chart = go.Bar(x = col_mean_count.head(10).index, y = col_mean_count.head(10).values, name="Mean",marker = dict(color=color2))

    fig = tools.make_subplots(rows = 1, cols = 2,subplot_titles=('Total Count','Mean Revenue'))
    fig.append_trace(col_count_chart, 1,1)
    fig.append_trace(col_mean_count_chart,1,2)
    py.iplot(fig)

train.head()
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
train_df = load_df()
train_df.head()
test_df = load_df("../input/test.csv")
# train_df.to_csv('flatten_train.csv')
# test_df.to_csv('flatten_test.csv')
diff_cols(train_df,test_df)
train_constants = constant_cols(train_df)
test_constants = constant_cols(test_df)
print(train_constants)
print(test_constants)
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].fillna(0)
train_df['date'] = train_df['date'].astype(str)
train_df["date"] = train_df["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
train_df["date"] = pd.to_datetime(train_df["date"])
train_constants = constant_cols(train_df)
test_constants = constant_cols(test_df)
train_df = train_df.drop(columns=train_constants,axis = 1)
test_df = test_df.drop(columns=test_constants, axis = 1)
null_values = train_df.isna().sum(axis = 0).reset_index()
null_values = null_values[null_values[0] > 50]
null_chart = [go.Bar(y = null_values['index'],x = null_values[0]*100/len(train_df), orientation = 'h')]
py.iplot(null_chart)
data = train_df[['channelGrouping','totals.transactionRevenue']]
temp = data['channelGrouping'].value_counts()
chart = [go.Pie(labels = temp.index, values = temp.values)]
py.iplot(chart)
temp = train_df['device.isMobile'].value_counts()
chart = go.Bar(x = ["False","True"], y = temp.values)
py.iplot([chart])
count_mean('device.browser',"#7FDBFF","#3D9970")
count_mean('device.deviceCategory',"#FF851B","#FF4136")
count_mean('device.operatingSystem',"#80DEEA","#0097A7")
count_mean('geoNetwork.continent',"#F48FB1","#C2185B")
data = train_df[['geoNetwork.country','totals.transactionRevenue']][(train_df['totals.transactionRevenue'] >1)]
temp = data.groupby('geoNetwork.country',as_index=False)['totals.transactionRevenue'].mean()
temp['code'] = 'sample'
for i,country in enumerate(temp['geoNetwork.country']):
    mapping = {country.name: country.alpha_3 for country in pycountry.countries}
    temp.set_value(i,'code',mapping.get(country))
chart = [ dict(
        type = 'choropleth',
        locations = temp['code'],
        z = temp['totals.transactionRevenue'],
        text = temp['geoNetwork.country'],
        autocolorscale = True,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                  width = 0.5
            ) ),
        colorbar = dict(
            autotick = True,
            title = 'Mean Revenue'),
      ) ]

layout = dict(
    title = 'Mean revenue based on country',
    geo = dict(
        showframe = True,
        showcoastlines = True,
         showocean = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=chart, layout=layout )
py.iplot( fig, validate=False)
count_mean('geoNetwork.metro',"#CE93D8", "#7B1FA2")
count_mean('geoNetwork.networkDomain','#90CAF9','#1976D2')
count_mean('geoNetwork.region','#DCE775','#AFB42B')
count_mean('geoNetwork.subContinent','#FFE082','#FFA000')
train_df['totals.pageviews'] = train_df['totals.pageviews'].fillna(0).astype('int32')
train_df['totals.bounces'] = train_df['totals.bounces'].fillna(0).astype('int32')

pageview = train_df.groupby('date')['totals.pageviews'].apply(lambda x:x[x >= 1].count()).reset_index()
bounce = train_df.groupby('date')['totals.bounces'].apply(lambda x:x[x >= 1].count()).reset_index()

pageviews = go.Scatter(x = pageview['date'],y= pageview['totals.pageviews'], name = 'Pageview',marker=dict(color = "#B0BEC5"))


bounces = go.Scatter(x = bounce['date'],y= bounce['totals.bounces'],name = 'Bounce',marker=dict(color = "#37474F"))

py.iplot([pageviews,bounces])
train_df['totals.newVisits'] = train_df['totals.newVisits'].fillna(0).astype('int32')
train_df['totals.hits'] = train_df['totals.hits'].fillna(0).astype('int32')

newvisit = train_df.groupby('date')['totals.newVisits'].apply(lambda x:x[x == 1].count()).reset_index()
oldVisit = train_df.groupby('date')['totals.newVisits'].apply(lambda x:x[x == 0].count()).reset_index()
hit = train_df.groupby('date')['totals.hits'].apply(lambda x:x[x >= 1].count()).reset_index()


hits = go.Scatter(x = hit['date'],y = hit['totals.hits'], name = 'total hits', marker=dict(color = '#FFEE58'))

new_vist = go.Scatter(x = newvisit['date'],y= newvisit['totals.newVisits'],name = 'New Vists', marker=dict(color = '#F57F17'))

oldvisit = go.Scatter(x = oldVisit['date'],y = oldVisit['totals.newVisits'], name = 'Old Visit', marker=dict(color = '#FFD600'))

py.iplot([hits, new_vist, oldvisit])
temp = train_df[(train_df['totals.transactionRevenue'] >0)]
data = temp[['totals.transactionRevenue','date']].groupby('date')['totals.transactionRevenue'].agg(['min','max']).reset_index()
mean = go.Scatter(x = data['date'], y = data['min'],name = "Min",marker = dict(color = '#00E676'))
count = go.Scatter(x = data['date'],y = data['max'], name = "Max",marker = dict(color = '#00838F'))
py.iplot([mean,count])
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['weekday'] = train_df['date'].dt.weekday
temp = train_df.groupby('month')['totals.transactionRevenue'].agg(['count','mean']).reset_index()
count_chart = go.Bar(x = temp['month'], y = temp['count'],name = 'Count',marker = dict(color = "#E6EE9C"))
mean_chart = go.Bar(x = temp['month'],y = temp['mean'], name = 'Mean',marker = dict(color = "#AFB42B"))

fig = tools.make_subplots(rows = 1, cols = 2, subplot_titles = ('Total Count', 'Mean Count'))
fig.append_trace(count_chart,1,1)
fig.append_trace(mean_chart, 1,2)
py.iplot(fig)
temp = train_df.groupby('day')['totals.transactionRevenue'].agg(['count','mean']).reset_index()
count_chart = go.Bar(x = temp['day'], y = temp['count'],name = 'Count', marker = dict(color = '#1DE9B6'))
mean_chart = go.Bar(x = temp['day'],y = temp['mean'], name = 'Mean', marker = dict(color = '#00796B'))

fig = tools.make_subplots(rows = 1, cols = 2, subplot_titles = ('Total Count', 'Mean Count'))
fig.append_trace(count_chart,1,1)
fig.append_trace(mean_chart, 1,2)
py.iplot(fig)
temp = train_df.groupby('weekday')['totals.transactionRevenue'].agg(['count','mean']).reset_index()
count_chart = go.Bar(x = temp['weekday'], y = temp['count'],name = 'Count', marker = dict(color = '#9575CD'))
mean_chart = go.Bar(x = temp['weekday'],y = temp['mean'], name = 'Mean', marker = dict(color = '#B388FF'))

fig = tools.make_subplots(rows = 1, cols = 2, subplot_titles = ('Total Count', 'Mean Count'))
fig.append_trace(count_chart,1,1)
fig.append_trace(mean_chart, 1,2)
py.iplot(fig)
train_df['trafficSource.adContent'] = train_df['trafficSource.adContent'].fillna('')
wordcloud2 = WordCloud(width=800, height=400).generate(' '.join(train_df['trafficSource.adContent']))
plt.figure( figsize=(15,20))
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()
train_df['trafficSource.keyword'] = train_df['trafficSource.keyword'].fillna('')
wordcloud2 = WordCloud(width=800, height=400).generate(' '.join(train_df['trafficSource.keyword']))
plt.figure( figsize=(20,20) )
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()
train_df['trafficSource.source'] = train_df['trafficSource.source'].fillna('')
wordcloud2 = WordCloud(width=800, height=400).generate(' '.join(train_df['trafficSource.source']))
plt.figure( figsize=(15,20) )
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()
