# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import json

import ast

import numpy as np

import pandas as pd

from pandas.io.json import json_normalize

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import seaborn as sns

plt.rcParams['axes.grid'] = True



from sklearn import model_selection, preprocessing, metrics

import lightgbm as lgb



np.random.seed(seed=45)







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!ls ../input/
def load_df(csv_path='../input/train_v2.csv',nrows=None):

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    

    df = pd.read_csv(csv_path, 

                     converters={column: json.loads for column in JSON_COLUMNS}, 

                     dtype={'fullVisitorId': 'str'},

                     nrows=nrows)

    print('nrows: %d',nrows)

    

    for column in JSON_COLUMNS:

        column_as_df = json_normalize(df[column])

        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]

        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")

    return df
%%time

train_df = load_df(nrows=3*pow(10,5))

#test_df = load_df("../input/test.csv")
train_df.shape
train_df.sample(n=5)
train_df.drop_duplicates(inplace=True)

train_df.shape
missing_df = (train_df.isna().sum()/train_df.shape[0])*100

missing_df = missing_df.loc[missing_df.values > 20]

missing_df
ratio_of_conversion = train_df['totals_transactionRevenue'].notnull().sum()/train_df.shape[0]

print('Only %f percent of visits are converted into revenue' % (ratio_of_conversion*100))
train_df.customDimensions.sample(n=10)
train_df.drop(columns='customDimensions', inplace=True)
train_df.head()
train_df.hits.sample(n=1, random_state=42).values
hits_dict = ast.literal_eval(train_df.hits.sample(n=1, random_state=42).values[0])[0]

display(hits_dict)
train_df.drop(columns='hits', inplace=True)
train_df.sample(n=5)
# Fill the NA values with 'nan' for tracking the nan values in groupby functions

train_df.fillna('nan',inplace=True)
# remove the missing columns we got

#train_df = train_df.drop(columns=missing_df.index)
# Find columns with only one unique value and drop them

constant_columns = [ c for c in train_df.columns if train_df[c].nunique(dropna=False)==1]

display(constant_columns)

train_df.drop(columns=constant_columns, inplace=True)
train_df.head()
train_df.dtypes
train_df.totals_bounces.describe()
train_df['totals_bounces'] = train_df.totals_bounces.apply(lambda x: 0 if x=='nan' else x)

train_df['totals_bounces'] = train_df['totals_bounces'].astype('int16')
train_df.totals_hits.describe()
train_df.totals_pageviews.describe()
train_df.totals_sessionQualityDim.describe()
# If a new visit is nan it can be 1 but then it becomes only one value, so we can drop this column

display(train_df['totals_newVisits'].unique())

train_df.drop(columns='totals_newVisits', inplace=True)
# I find it safe to put 0 for nan values because those could be bots that visits the website

display(train_df['totals_timeOnSite'].describe())

print('Top 5 most common values for totals_timeOnSite: ')

display(train_df.groupby('totals_timeOnSite').size().sort_values(ascending=False)[:5])

train_df['totals_timeOnSite'] = train_df.totals_timeOnSite.apply(lambda x: 0 if x=='nan' else x)

train_df['totals_timeOnSite'] = train_df['totals_timeOnSite'].astype('int64')
# A minimum hit in a visit can be 1 and 1 is also the mode number

train_df['totals_hits'] = train_df.totals_hits.apply(lambda x: 1 if x=='nan' else x)

train_df['totals_hits'] = train_df['totals_hits'].astype('int64')

# minimum pageview can be 1 for a visit and mode is also 1

train_df['totals_pageviews'] = train_df.totals_pageviews.apply(lambda x: 1 if x=='nan' else x)

train_df['totals_pageviews'] = train_df['totals_pageviews'].astype('int64')

# sessionQuality represents a score between 1-100 that shows how close is the user to the conversion to a transaction in the website

train_df['totals_sessionQualityDim'] = train_df.totals_sessionQualityDim.apply(lambda x: 1 if x=='nan' else x)

train_df['totals_sessionQualityDim'] = train_df['totals_sessionQualityDim'].astype('int16')

# When transaction variables are 'nan' we considered them transaction has not been made by the user, so 0 is the appropriate number

train_df['totals_totalTransactionRevenue'] = train_df.totals_totalTransactionRevenue.apply(lambda x: 0 if x=='nan' else x)

train_df['totals_totalTransactionRevenue'] = train_df['totals_totalTransactionRevenue'].astype('int64')

train_df['totals_transactionRevenue'] = train_df.totals_transactionRevenue.apply(lambda x: 0 if x=='nan' else x)

train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].astype('int64')

train_df['totals_transactions'] = train_df.totals_transactions.apply(lambda x: 0 if x=='nan' else x)

train_df['totals_transactions'] = train_df['totals_transactions'].astype('int64')
# We can set 'nan' to None which represents a direct connection to website, not through google adWords then convert it to a category variable

# as per its stated https://support.google.com/analytics/answer/3437719?hl=en

display(train_df['trafficSource_adwordsClickInfo.adNetworkType'].unique())

train_df['trafficSource_adwordsClickInfo.adNetworkType'] = train_df['trafficSource_adwordsClickInfo.adNetworkType'].apply(lambda x: 'None' if x=='nan' else x)

# rename the column for easy reference

train_df.rename(columns={'trafficSource_adwordsClickInfo.adNetworkType':'trafficSource_adwords_adNetworkType'}, inplace=True)
# Fix the adContent variable, by converting 'nan' to None since user visits website without going through an ad campaign

display(train_df['trafficSource_adContent'].describe())

adContent_series = train_df.groupby('trafficSource_adContent').size().sort_values(ascending=False)

most_common_sources = adContent_series[adContent_series.cumsum() < train_df.shape[0]*0.95].index

train_df.trafficSource_adContent = train_df.trafficSource_adContent.apply(lambda x: x if x in most_common_sources else 'others')

train_df.trafficSource_adContent.describe()

train_df.drop(columns='trafficSource_adContent', inplace=True)
# trafficSource_adwordsClickInfo.gclId represents google click ID, we can drop it as it is not related with revenue generation and it is not

# easily interpreted by humans

display(train_df['trafficSource_adwordsClickInfo.gclId'].unique())

train_df.drop(columns='trafficSource_adwordsClickInfo.gclId', inplace=True)
# trafficSource_adwordsClickInfo.isVideoAd shows whether the user is coming from a video ad

display(train_df['trafficSource_adwordsClickInfo.isVideoAd'].unique())
# converting the name of the column for convenience and convert 'nan' to Unknown

train_df.rename(columns={'trafficSource_adwordsClickInfo.isVideoAd':'trafficSource_adwords_isVideoAd'}, inplace=True)

train_df['trafficSource_adwords_isVideoAd'] = train_df.trafficSource_adwords_isVideoAd.apply(lambda x: 'Unknown' if x=='nan' else x)
# This column is an ordinal variable, we will replace 'nan' with -1

display(train_df['trafficSource_adwordsClickInfo.page'].unique())

train_df.rename(columns={'trafficSource_adwordsClickInfo.page':'trafficSource_adwordsClickInfo_page'}, inplace=True)

train_df['trafficSource_adwordsClickInfo_page'] = train_df.trafficSource_adwords_isVideoAd.apply(lambda x: 'Not given' if x=='nan' else x)
train_df.drop(columns=['trafficSource_adwordsClickInfo.slot','trafficSource_campaignCode'], inplace=True)
# too many unique keywords exist and 95% is missing -> dropping the column

display(train_df['trafficSource_keyword'].describe())

train_df.drop(columns='trafficSource_keyword',inplace=True)
# since there are too many unique values and more than 60% of the data is missing, I am removing this column

display(train_df['trafficSource_referralPath'].unique())

display(train_df['trafficSource_referralPath'].describe())

train_df.drop(columns='trafficSource_referralPath',inplace=True)
# there are too many unique values for a categorical variable we need to decrease this number by making a histogram of most common sources

display(train_df['trafficSource_source'].describe())
sources_series = (train_df.groupby('trafficSource_source').size().sort_values()/train_df.shape[0])*100



trace1 = go.Bar(

            x=sources_series.values,

            y=sources_series.index,

            marker = dict(color = 'rgba(100, 155, 15, 1.0)'),

            orientation = 'h',

            name='percentage'

)





layout = go.Layout(title='% representation of each unique trafficSource category',

                   height=3000,

                   width=1000,

                   margin = dict(l=300,r=100,t=140, b=80),

                   xaxis=dict(title='categories'),

                   yaxis=dict(title='percentage')

)



fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig, filename='Revenue generation with pageviews')
# By analysis of the above plot, categories covering 95% of all sessions considered enough for representation

most_common_sources = sources_series[sources_series.cumsum() < train_df.shape[0]*0.98].index

train_df.trafficSource_source = train_df.trafficSource_source.apply(lambda x: x if x in most_common_sources else 'others')

train_df.trafficSource_source.describe()


display(train_df.trafficSource_isTrueDirect.unique())

train_df.trafficSource_isTrueDirect = train_df.trafficSource_isTrueDirect.apply(lambda x: 'Unknown' if x == 'nan' else x)
cities_series = (train_df.groupby('geoNetwork_city').size().sort_values()/train_df.shape[0])*100



trace1 = go.Bar(

            x=cities_series.values,

            y=cities_series.index,

            marker = dict(color = 'rgba(100, 175, 115, 1.0)'),

            orientation = 'h',

            name='percentage'

)





layout = go.Layout(title='% representation of each unique trafficSource category',

                   height=3000,

                   width=1000,

                   margin = dict(l=300,r=100,t=140, b=80),

                   xaxis=dict(title='categories'),

                   yaxis=dict(title='percentage')

)



fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig, filename='Revenue generation with pageviews')
cities_series = train_df.groupby('geoNetwork_city').size().sort_values(ascending=False)

most_common_cities = cities_series[cities_series.cumsum() < train_df.shape[0]*0.80].index

train_df.geoNetwork_city = train_df.geoNetwork_city.apply(lambda x: x if x in most_common_cities else 'others')

display(train_df['geoNetwork_city'].nunique())
# Nominal variables

train_df['channelGrouping'] = train_df['channelGrouping'].astype('category')

train_df['device_browser'] = train_df['device_browser'].astype('category')

train_df['device_deviceCategory'] = train_df['device_deviceCategory'].astype('category')

train_df['device_operatingSystem'] = train_df['device_operatingSystem'].astype('category')

train_df['geoNetwork_city'] = train_df['geoNetwork_city'].astype('category')

train_df['geoNetwork_continent'] = train_df['geoNetwork_continent'].astype('category')

train_df['geoNetwork_metro'] = train_df['geoNetwork_metro'].astype('category')

train_df['geoNetwork_region'] = train_df['geoNetwork_region'].astype('category')

train_df['geoNetwork_subContinent'] = train_df['geoNetwork_subContinent'].astype('category')

train_df['trafficSource_medium'] = train_df['trafficSource_medium'].astype('category')

train_df['trafficSource_adwords_adNetworkType'] = train_df['trafficSource_adwords_adNetworkType'].astype('category')

train_df['trafficSource_adwords_isVideoAd'] = train_df['trafficSource_adwords_isVideoAd'].astype('category')

train_df['trafficSource_source'] = train_df['trafficSource_source'].astype('category')

train_df['trafficSource_isTrueDirect'] = train_df['trafficSource_isTrueDirect'].astype('category')
train_df.sample(n=5)
# fix the date format

import datetime

# convert string formatted date to datetime object

train_df['date'] = train_df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))

# we still use to_datetime so that pandas can know it is Datetime

train_df['date'] = pd.to_datetime(train_df['date'],yearfirst=True)
# set an index column

#train_df.set_index(['fullVisitorId', 'visitId'], inplace=True)

#train_df.reset_index(inplace=True)
train_df.sample(n=5)
train_df.rename(columns={'device_browser': 'browser', 'device_deviceCategory':'deviceCategory', 'device_isMobile':'isMobile',

                        'device_operatingSystem': 'operatingSystem', 'geoNetwork_city':'city', 'geoNetwork_continent': 'continent',

                        'geoNetwork_country':'country', 'geoNetwork_metro':'metro', 'geoNetwork_networkDomain':'networkDomain',

                        'geoNetwork_region':'region', 'totals_bounces':'bounces', 'totals_hits':'hits', 'totals_pageviews':'pageviews',

                        'totals_sessionQualityDim':'sessionQuality', 'totals_timeOnSite':'timeOnSite', 'totals_totalTransactionRevenue':'totalTransactionRevenue',

                        'totals_transactionRevenue':'transactionRevenue', 'totals_transactions':'transactions', 'trafficSource_adwords_adNetworkType':'adwords_adNetworkType',

                        'trafficSource_adwords_isVideoAd':'adwords_isVideoAd', 'trafficSource_adwordsClickInfo_page': 'adwords_pageNumber'}, inplace=True)
train_df.head()
train_df["transactionRevenue"] = train_df["transactionRevenue"].astype("float")

gdf = train_df.groupby(["fullVisitorId"])["transactionRevenue"].sum().reset_index()

display(gdf.sample(5))

plt.figure(figsize= (8,6))

plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["transactionRevenue"].values)))

plt.xlabel('index', fontsize=12)

plt.ylabel('TransactionRevenue', fontsize=12)

plt.show()
nzi = pd.notnull(train_df["transactionRevenue"]).sum()

nzr = (gdf["transactionRevenue"]>0).sum()

print("Number of instances in train set with non-zero revenue : ", nzi, " and ratio is : ", nzi / train_df.shape[0])

print("Number of unique customers with non-zero revenue : ", nzr, "and the ratio is : ", nzr / gdf.shape[0])
# According to 

train_df['transactionRevenue'].describe()
plt.figure(figsize=(12,6))

# I tried to scale the transactionRevenue variable to be able to get a distribution of transactions' values. 

# I will not use this as a label as it squeezes the distribution

# Why divide by 10**6 check: https://support.google.com/analytics/answer/3437719?hl=en

y = train_df['transactionRevenue']/10**6

mask = (y>0) & (y < 1000)

revenue_df = y[mask]





n, bins, patches = plt.hist(revenue_df.values, 100, facecolor='b', alpha=0.75)

plt.xticks(rotation='horizontal')

plt.xlabel('Transaction revenue', fontsize=12)

plt.ylabel('Number of Transactions', fontsize=12)

plt.show()
mask = (y > 1000)

total_revenue_1000 = y[mask].sum()

total_revenue = y.sum()



data_dict = {'Number of transactions over 1000 USD revenues': mask.sum(),

             'Percent total revenue' : [np.round(total_revenue_1000/total_revenue*100,2)]}

tail_df = pd.DataFrame(data=data_dict)

tail_df
def scatter_plot(cnt_srs, color, name):

    trace = go.Scatter(

        x=cnt_srs.index[::-1],

        y=cnt_srs.values[::-1],

        showlegend=False,

        marker=dict(

            color=color,

        ),

        name = name

    )

    return trace





trans_per_day = train_df.groupby('date')['transactionRevenue'].agg(['sum','count'])

trans_per_day.columns = ["revenue", "session count"]

trans_per_day = trans_per_day.sort_index()





non_zero_trans = train_df[train_df['transactionRevenue']>0]

non_zero_trans_per_day = non_zero_trans.groupby('date')['transactionRevenue'].agg(['count'])

non_zero_trans_per_day.columns = ["count of non-zero revenue"]

non_zero_trans_per_day = non_zero_trans_per_day.sort_index()





trace1 = scatter_plot(trans_per_day["revenue"]/10**6, 'red','Session count')

trace2 = scatter_plot(trans_per_day["session count"], 'blue','Session count')

trace3 = scatter_plot(non_zero_trans_per_day["count of non-zero revenue"], 'green','Session count')



fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.08, horizontal_spacing=0.08,

                          subplot_titles=["Date - Revenue", "Date - Session count", "Date - Non-zero revenue"])

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 2, 1)

fig.append_trace(trace3, 3, 1)

fig['layout'].update(height=1000, width=1000,title="Date Plots")

fig['layout']['xaxis1'].update(title='dates')

fig['layout']['xaxis2'].update(title='dates')

fig['layout']['xaxis3'].update(title='dates')



py.iplot(fig, filename='date-plots')
# Check date has any NaNs

train_df['date'].isnull().any()
# First convert date to weekdays

train_df['weekday'] = train_df['date'].dt.day_name()

train_df['weekday'].unique()
session_df = train_df.groupby('weekday')['fullVisitorId'].nunique()

revenue_df = train_df.groupby('weekday')['transactionRevenue'].sum()/10**6





plt.figure(figsize=(12,4))

#plot the transaction distribution by month

sns.barplot(session_df.index, session_df.values, alpha=0.8, color=color[2])

plt.xticks(rotation='vertical')

plt.xlabel('Weekday of sessions', fontsize=12)

plt.ylabel('Number of sessions', fontsize=12)

plt.show()



plt.figure(figsize=(12,4))

sns.barplot(revenue_df.index, revenue_df.values, alpha=0.8, color=color[6])

plt.xticks(rotation='vertical')

plt.xlabel('Weekday of sessions', fontsize=12)

plt.ylabel('Total Revenue', fontsize=12)

plt.show()

# First convert date to Months

train_df['month'] = train_df['date'].dt.month_name()

train_df['month'].unique()
session_df = train_df.groupby('month')['fullVisitorId'].nunique()

revenue_df = train_df.groupby('month')['transactionRevenue'].sum()/10**6





plt.figure(figsize=(12,4))

#plot the transaction distribution by month

sns.barplot(session_df.index, session_df.values, alpha=0.8, color=color[1])

plt.xticks(rotation='vertical')

plt.xlabel('Month of sessions', fontsize=12)

plt.ylabel('Number of sessions', fontsize=12)

plt.show()



plt.figure(figsize=(12,4))

sns.barplot(revenue_df.index, revenue_df.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Month of sessions', fontsize=12)

plt.ylabel('Total Revenue', fontsize=12)

plt.show()
#First check if theres any null for channelGrouping

train_df['channelGrouping'].isnull().any()
session_df = train_df.groupby('channelGrouping')['fullVisitorId'].nunique()

session_df.sort_values(ascending=False, inplace=True)



revenue_df = train_df.groupby('channelGrouping')['transactionRevenue'].sum()/10**6



plt.figure(figsize=(12,4))

#plot the transaction distribution by month

sns.barplot(session_df.index, session_df.values, alpha=0.8, color='dodgerblue')

plt.xticks(rotation='horizontal')

plt.xlabel('Channel Groupings', fontsize=12)

plt.ylabel('Number of sessions', fontsize=12)

plt.show()



plt.figure(figsize=(12,4))

#plot the transaction distribution by month

sns.barplot(revenue_df.index, revenue_df.values, alpha=0.8, color='yellowgreen')

plt.xticks(rotation='horizontal')

plt.xlabel('Channel Groupings', fontsize=12)

plt.ylabel('Total Revenue', fontsize=12)

plt.show()
session_df = train_df.groupby('deviceCategory')['fullVisitorId'].nunique()

revenue_df = train_df.groupby('deviceCategory')['transactionRevenue'].sum()/10**6





data1 = go.Bar(

            x=session_df.values,

            y=session_df.index,

            orientation = 'h',

            marker = dict(color = 'rgba(246, 78, 139, 1.0)'),

            name='session count'

)



data2 = go.Bar(

            x=revenue_df.values,

            y=revenue_df.index,

            orientation = 'h',

            name='revenue(in dollars)'

)



fig = tools.make_subplots(rows=1, cols=2)

fig['layout'].update(height=400, width=800, title='Session and Revenue generation per device segment')

fig['layout']['xaxis1'].update(title='Session Count')

fig['layout']['xaxis2'].update(title='Revenue (in USD)')



fig.append_trace(data1, 1, 1)

fig.append_trace(data2, 1, 2)

py.iplot(fig, filename='simple-subplot-with-annotations')
session_df = train_df.groupby('operatingSystem')['fullVisitorId'].nunique()

revenue_df = train_df.groupby('operatingSystem')['transactionRevenue'].sum()/10**6



# sort and pick top 7

session_df = session_df[session_df.index != "(not set)"]

session_df = session_df.sort_values(ascending=False)[:7]

display(session_df.index.values)



revenue_df = revenue_df[revenue_df.index != "(not set)"]

revenue_df = revenue_df.sort_values(ascending=False)[:7]





data1 = go.Bar(

            x=session_df.values,

            y=session_df.index,

            orientation = 'h',

            marker = dict(color = 'rgba(246, 78, 25, 1.0)'),

            name='session count'

)



data2 = go.Bar(

            x=revenue_df.values,

            y=revenue_df.index,

            orientation = 'h',

            marker = dict(color = 'rgba(115, 20, 139, 1.0)'),

            name='revenue'

)



fig = tools.make_subplots(rows=1, cols=2)

fig['layout'].update(height=400, width=800, title='Session and Revenue generation per OS')

fig['layout']['xaxis1'].update(title='Session Count')

fig['layout']['xaxis2'].update(title='Total Revenue (in USD)')



fig.append_trace(data1, 1, 1)

fig.append_trace(data2, 1, 2)

py.iplot(fig, filename='Session and Revenue generation per OS')



browser_df = train_df.groupby('browser')['fullVisitorId'].nunique()

browser_df = browser_df.sort_values(ascending=False)[:15]



browser_revenue_df = train_df.groupby('browser')['transactionRevenue'].sum()/10**6

browser_revenue_df = browser_revenue_df.sort_values(ascending=False)[:15]



data1 = go.Bar(

            x=browser_df.values,

            y=browser_df.index,

            orientation = 'h',

            marker = dict(color = 'rgba(225, 55, 25, 1.0)'),

            name='session count'

)



data2 = go.Bar(

            x=browser_revenue_df.values,

            y=browser_revenue_df.index,

            orientation = 'h',

            marker = dict(color = 'rgba(200, 20, 139, 1.0)'),

            name='revenue'

)



fig = tools.make_subplots(rows=1, cols=2)

fig['layout'].update(height=800, width=1000, title='Session and Revenue generation per Browser')

fig['layout']['xaxis1'].update(title='Session Count')

fig['layout']['xaxis2'].update(title='Total Revenue (in USD)')



fig.append_trace(data1, 1, 1)

fig.append_trace(data2, 1, 2)

py.iplot(fig, filename='Session and Revenue generation per OS')

revenue_df = train_df.groupby('pageviews')['transactionRevenue'].sum()/10**6

revenue_df = revenue_df.sort_values(ascending=False)



data1 = go.Bar(

            x=revenue_df.index,

            y=revenue_df.values,

            marker = dict(color = 'rgba(225, 55, 25, 1.0)'),

            name='revenue'

)



layout = go.Layout(title='Revenue generation with pageviews',

                   height=400,

                   width=800,

                   xaxis=dict(title='Pageviews'),

                   yaxis=dict(title='Revenue (in USD)')

)



fig = go.Figure(data=[data1], layout=layout)

py.iplot(fig, filename='Revenue generation with pageviews')
train_df.head()
top_k_countries = 20

country_df = train_df.groupby('country')['fullVisitorId'].nunique()

country_df = country_df.sort_values(ascending=False)[:top_k_countries]

country_revenue_df = train_df.groupby('country')['transactionRevenue'].sum()/10**6

country_revenue_df = country_revenue_df.sort_values(ascending=False)[:top_k_countries]



trace1 = go.Bar(

            x=country_df.values,

            y=country_df.index,

            marker = dict(color = 'rgba(55, 55, 125, 1.0)'),

            orientation = 'h',

            name='session count'

)



trace2 = go.Bar(

            x=country_revenue_df.values,

            y=country_revenue_df.index,

            marker = dict(color = 'rgba(55, 155, 25, 1.0)'),

            orientation = 'h',

            name='revenue'

)



fig = tools.make_subplots(rows=1, cols=2)

fig['layout'].update(height=800, width=1000, title='Session and Revenue generation per Country')

fig['layout']['margin'].update(l=120,r=10,t=140, b=80 )

fig['layout']['xaxis1'].update(title='Session Count')

fig['layout']['xaxis2'].update(title='Total Revenue (in USD)')



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

py.iplot(fig, filename='Session and Revenue generation per Country')
train_df.transactions.describe()
session_quality_hist = train_df[['sessionQuality','transactionRevenue']]

session_quality_hist = session_quality_hist[session_quality_hist['transactionRevenue'] > 0].sessionQuality.value_counts()

session_quality_df = train_df.groupby('sessionQuality').size()

session_quality_df = session_quality_df.sort_values(ascending=False)

session_quality_revenue_df = train_df.groupby('sessionQuality')['transactionRevenue'].sum()/10**6

session_quality_revenue_df = session_quality_revenue_df.sort_values(ascending=False)



trace1 = go.Bar(

            x=session_quality_hist.index,

            y=session_quality_hist.values,

            marker = dict(color = 'rgba(55, 72, 155, 1.0)'),

            name='session count'

)



trace0 = go.Bar(

            x=session_quality_df.index,

            y=session_quality_df.values,

            marker = dict(color = 'rgba(155, 55, 125, 1.0)'),

            name='session count'

)



trace2 = go.Bar(

            x=session_quality_revenue_df.index,

            y=session_quality_revenue_df.values,

            marker = dict(color = 'rgba(55, 100, 25, 1.0)'),

            name='revenue'

)



fig = tools.make_subplots(rows=3, cols=1)

fig['layout'].update(height=1000, width=1500, title='Session and Revenue generation with SessionQuality')

fig['layout']['margin'].update(l=120,r=250,t=140, b=80 )

fig['layout']['yaxis2'].update(title='Non-zero revenue session count')

fig['layout']['xaxis1'].update(title='SessionQuality (1-100)')

fig['layout']['yaxis1'].update(title='Total Sessions')

fig['layout']['xaxis2'].update(title='SessionQuality (1-100)')

fig['layout']['yaxis3'].update(title='Total Revenue (in USD)')

fig['layout']['xaxis3'].update(title='SessionQuality (1-100)')



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 2, 1)

fig.append_trace(trace2, 3, 1)

py.iplot(fig, filename='Session and Revenue generation with SessionQuality')



time_spend_df = train_df.groupby('timeOnSite')['transactionRevenue'].sum()/10**6

time_spend_df = time_spend_df.sort_values(ascending = False)

sorted_times = train_df['timeOnSite']



trace1 = go.Bar(

            x=time_spend_df.index,

            y=time_spend_df.values,

            marker = dict(color = 'rgba(175, 55, 25, 1.0)'),

            name='revenue'

)



trace2= go.Histogram(

            histnorm='percent',

            marker = dict(color = 'rgba(100, 155, 25, 0.7)'),

            x = sorted_times.values,

            xbins=dict(

                start=0,

                end=3000,

                size=30

            ),

            name='percentage of sessions'

)

        



fig = tools.make_subplots(rows=2, cols=1)

fig['layout'].update(height=1000, width=1500, title='Session and Revenue generation with timeOnSite')

fig['layout']['margin'].update(l=120,r=250,t=140, b=80 )

fig['layout']['yaxis1'].update(title='Revenue (USD)')

fig['layout']['xaxis1'].update(title='Time spent on site (in secs)')

fig['layout']['yaxis2'].update(title='Percentage of Sessions (in %)')

fig['layout']['xaxis2'].update(title='Time spent on site (in secs)')



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 2, 1)

py.iplot(fig, filename='Session and Revenue generation with SessionQuality')



train_df.head()
video_ads_df = train_df.groupby('adwords_isVideoAd')['transactionRevenue'].sum()/10**6



# realize that most of the videoAd column is unknown / we may get rid of this column doesn't provide useful information

display(train_df.adwords_isVideoAd.describe())



trace1 = go.Bar(

            x=video_ads_df.values,

            y=video_ads_df.index,

            marker = dict(color = 'rgba(175, 155, 25, 1.0)'),

            orientation='h',

            name='Revenue'

)





layout = go.Layout(title='Revenue generation with video ads',

                   height=400,

                   width=800,

                   xaxis=dict(title='Revenue (in USD)'),

                   yaxis=dict(title='IsVideoAd')

)



fig = go.Figure(data=[trace1], layout=layout)

py.iplot(fig, filename='Revenue generation with pageviews')
region_df = train_df.groupby('visitNumber')['transactionRevenue'].sum()/10**6





trace1 = go.Bar(

            x=region_df.index,

            y=region_df.values,

            marker = dict(color = 'rgba(200, 55, 25, 1.0)'),

            name='revenue'

)



trace2= go.Histogram(

            histnorm='percent',

            marker = dict(color = 'rgba(100, 155, 215, 0.7)'),

            x = train_df['visitNumber'].values,

            xbins=dict(

                start=0,

                end=1000,

                size=2

            ),

            name='percentage of sessions'

)

        



fig = tools.make_subplots(rows=2, cols=1)

fig['layout'].update(height=1000, width=1000, title='Session percentages and Revenue generation with VisitCounts')

fig['layout']['margin'].update(l=120,r=25,t=140, b=80 )

fig['layout']['yaxis1'].update(title='Revenue (USD)')

fig['layout']['xaxis1'].update(title='visit count to the site')

fig['layout']['yaxis2'].update(title='Percentage of Sessions (in %)')

fig['layout']['xaxis2'].update(title='visit count to the site')



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 2, 1)

py.iplot(fig, filename='Session and Revenue generation with VisitCounts')
# calculating number of sessions, non-zero revenue count and mean revenue from each geolocation and domain



def horizontal_bar_chart(series, color, name):

    trace = go.Bar(

        y=series.index,

        x=series.values,

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

        name = name

    )

    return trace





nz_revenue_mask = train_df['transactionRevenue'] > 0

continent_srs = train_df.groupby('continent')['transactionRevenue'].agg(['size','mean'])

continent_srs.columns = ['number of sessions', 'mean revenue']

continent_srs.sort_values(by='number of sessions', ascending=False, inplace=True)

continent_nz_revenue_srs = train_df[nz_revenue_mask].groupby('continent')['transactionRevenue'].size()

continent_nz_revenue_srs.sort_values(ascending=False, inplace=True)



subContinent_srs = train_df.groupby('geoNetwork_subContinent')['transactionRevenue'].agg(['size','mean'])

subContinent_srs.columns = ['number of sessions', 'mean revenue']

subContinent_srs.sort_values(by='number of sessions', ascending=False, inplace=True)

subContinent_nz_revenue = train_df[nz_revenue_mask].groupby('geoNetwork_subContinent')['transactionRevenue'].size()

subContinent_nz_revenue.sort_values(ascending=False, inplace=True)



networkDomain_srs = train_df.groupby('networkDomain')['transactionRevenue'].agg(['size','mean'])

networkDomain_srs.columns = ['number of sessions', 'mean revenue']

networkDomain_srs.sort_values(by='number of sessions', ascending=False, inplace=True)

networkDomain_nz_revenue_srs = train_df.loc[nz_revenue_mask].groupby('networkDomain')['transactionRevenue'].size()

networkDomain_nz_revenue_srs.sort_values(ascending=False, inplace=True)







trace1 = horizontal_bar_chart(continent_srs["number of sessions"].head(10), 'rgba(50, 171, 96, 0.6)',name='count')

trace2 = horizontal_bar_chart((continent_srs["mean revenue"]/10**6).head(10), 'rgba(50, 171, 96, 0.6)','revenue')

trace3 = horizontal_bar_chart(continent_nz_revenue_srs, 'rgba(50, 171, 96, 0.6)','count')





trace4 = horizontal_bar_chart(subContinent_srs["number of sessions"], 'rgba(71, 58, 131, 0.8)','count')

trace5 = horizontal_bar_chart(subContinent_srs["mean revenue"]/10**6, 'rgba(71, 58, 131, 0.8)','revenue')

trace6 = horizontal_bar_chart(subContinent_nz_revenue.head(10), 'rgba(71, 58, 131, 0.8)','count')



trace7 = horizontal_bar_chart(networkDomain_srs["number of sessions"].head(10), 'rgba(246, 78, 139, 0.6)','count')

trace8 = horizontal_bar_chart((networkDomain_srs["mean revenue"]/10**6).head(10), 'rgba(246, 78, 139, 0.6)','revenue')

trace9 = horizontal_bar_chart(networkDomain_nz_revenue_srs.head(10), 'rgba(246, 78, 139, 0.6)','count')



fig = tools.make_subplots(rows=3, cols=3, vertical_spacing=0.04, 

                          subplot_titles=["Continent - Count", "Continent - Mean Revenue", "Continent - Non-zero Revenue Count",

                                          "Subcontinent - Count",  "Subcontinent - Mean Revenue", "Subcontinent - Non-zero Revenue Count", 

                                          "NetworkDomain - Count", "NetworkDomain - Mean Revenue", "NetworkDomain - Non-zero Revenue Count"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.append_trace(trace4, 2, 1)

fig.append_trace(trace5, 2, 2)

fig.append_trace(trace6, 2, 3)

fig.append_trace(trace7, 3, 1)

fig.append_trace(trace8, 3, 2)

fig.append_trace(trace9, 3, 3)



fig['layout'].update(height=1200, width=1500, paper_bgcolor='rgb(233,233,233)',title="Geolocation and NetworkDomain Plots")

fig['layout']['margin'].update(l=100,r=25,t=75, b=80)

py.iplot(fig, filename='geolocation-plots')
pageviews_srs = train_df.groupby('pageviews')['transactionRevenue'].agg(['size','mean'])

pageviews_srs.columns = ['number of sessions','mean revenue']

pageviews_srs.sort_values(by='number of sessions', inplace=True,ascending=False)

pageviews_non_zero_revenue_srs = train_df[nz_revenue_mask].groupby('pageviews')['transactionRevenue'].size()

pageviews_non_zero_revenue_srs.sort_values(inplace=True, ascending=False)





hits_srs = train_df.groupby('hits')['transactionRevenue'].agg(['size','mean'])

hits_srs.columns = ['number of sessions','mean revenue']

hits_srs.sort_values(by='number of sessions', inplace=True, ascending=False)

hits_non_zero_revenue_srs = train_df[nz_revenue_mask].groupby('hits')['transactionRevenue'].size()

hits_non_zero_revenue_srs.sort_values(inplace=True, ascending=False)





trace1 = horizontal_bar_chart(pageviews_srs["number of sessions"].head(50), 'rgba(50, 171, 96, 0.6)',name='count')

trace2 = horizontal_bar_chart((pageviews_srs["mean revenue"]/10**6).head(50), 'rgba(50, 171, 96, 0.6)','dollars')

trace3 = horizontal_bar_chart(pageviews_non_zero_revenue_srs.head(50), 'rgba(50, 171, 96, 0.6)','count')





trace4 = horizontal_bar_chart(hits_srs["number of sessions"].head(50), 'rgba(71, 58, 131, 0.8)','count')

trace5 = horizontal_bar_chart((hits_srs["mean revenue"]/10**6).head(50), 'rgba(71, 58, 131, 0.8)','dollars')

trace6 = horizontal_bar_chart(hits_non_zero_revenue_srs.head(50), 'rgba(71, 58, 131, 0.8)','count')







fig = tools.make_subplots(rows=2, cols=3, vertical_spacing=0.04, 

                          subplot_titles=["Pageviews - Count", "Pageviews - Mean Revenue", "Pageviews - Non-zero Revenue Count",

                                          "Hits - Count",  "Hits - Mean Revenue", "Hits - Non-zero Revenue Count"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig.append_trace(trace4, 2, 1)

fig.append_trace(trace5, 2, 2)

fig.append_trace(trace6, 2, 3)



fig['layout'].update(height=1200, width=1500, paper_bgcolor='rgb(233,233,233)',title="Geolocation and NetworkDomain Plots")

fig['layout']['margin'].update(l=100,r=25,t=75, b=80)

py.iplot(fig, filename='geolocation-plots')
