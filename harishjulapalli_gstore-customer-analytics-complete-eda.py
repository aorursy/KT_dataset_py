import numpy as np

import pandas as pd

from pandas.io.json import json_normalize

import matplotlib.pyplot as plt

import seaborn as sns

import json



import plotly.graph_objs as go

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



%matplotlib inline

plt.style.use('ggplot')



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/train_v2.csv", nrows = 10000)

df.head()
def load_df(csv_path='../input/train_v2.csv', nrows=None):

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    

    df = pd.read_csv(csv_path, 

                     converters={column: json.loads for column in JSON_COLUMNS}, 

                     dtype={'fullVisitorId': 'str'}, # Important!!

                     nrows=nrows)

    

    for column in JSON_COLUMNS:

        column_as_df = json_normalize(df[column])

        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]

        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")

    return df

        
train = load_df(nrows = 200000)

train.head()
test = load_df(csv_path='../input/test_v2.csv', nrows = 50000)

test.head()
train.drop(['customDimensions','hits'], axis = 1, inplace = True)

#train.head()
test.drop(['customDimensions','hits'], axis = 1, inplace = True)
single_cat_cols = [col for col in train.columns if train.nunique()[col] == 1]
train.drop(single_cat_cols, axis =1, inplace = True)
single_cat_cols_test = [col for col in test.columns if test.nunique()[col] == 1]

test.drop(single_cat_cols_test , axis =1, inplace = True)
train.head()
print(f"Number of unique customers: {train.fullVisitorId.nunique()}")
print(f"Number of customers who visited the site more than once: {train.shape[0]-train.fullVisitorId.nunique()}")
train['totals_transactionRevenue'] = train['totals_transactionRevenue'].astype('float')
train['totals_transactionRevenue'].fillna(0, inplace = True)
train['totals_transactionRevenue'].sum()
reven_group = train.groupby('fullVisitorId')['totals_transactionRevenue'].sum().reset_index()
plt.figure(figsize = (8,6))

plt.scatter(range(reven_group.shape[0]), np.sort(np.log1p(reven_group['totals_transactionRevenue'])))

plt.xlabel('Index', fontsize = 14)

plt.ylabel('Log Transaction Revenue', fontsize = 14)

plt.title("Non-zero Revenue group")

plt.show()
print(f"Percentage of Revenue generating customers: {(train[train['totals_transactionRevenue'] > 0].shape[0]/train.shape[0])*100}")
#Pie-chart



labels = ['Non Revenue generating customers', 'Revenue generating customers']

non_revenue_perc = (train[train['totals_transactionRevenue'] == 0].shape[0]/train.shape[0])*100

revenue_perc = (train[train['totals_transactionRevenue'] > 0].shape[0]/train.shape[0])*100

sizes = [non_revenue_perc, revenue_perc]

explode = (0, 1)



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode = explode, labels = labels, autopct='%0.2f%%', colors = ['cornflowerblue','coral'], radius = 3)

plt.figure(figsize=(10,10))

ax1.axis('equal')

plt.tight_layout()

plt.show()
#Distribution of revenue generated

rg = train[train['totals_transactionRevenue'] > 0]['totals_transactionRevenue']


plt.figure(figsize = (8,6))

sns.distplot(rg)

plt.xlabel("Total Transaction Revenue", fontsize = 14)

plt.title("Distribution of Total Transaction Revenue")

plt.show()
plt.figure(figsize = (8,6))

sns.distplot(rg.apply(np.log1p))

plt.xlabel("Log Total Transaction Revenue", fontsize = 14)

plt.title("Distribution of Log Total Transaction Revenue")

plt.show()
from scipy.stats import kurtosis, skew

print(f"Skewness of transction value: {skew(rg.apply(np.log1p))}")

print(f"Kurtosis of transction value: {kurtosis(rg.apply(np.log1p))}")

plt.figure(figsize = (8,6))

sns.countplot(y = train['channelGrouping'], order = train['channelGrouping'].value_counts().index, palette='Blues_d')

plt.xlabel("Count", fontsize = 14)

plt.ylabel("Channel Grouping",fontsize = 14)

plt.title("Channel Grouping")

plt.show()
rg = train[train['totals_transactionRevenue'] > 0]



plt.figure(figsize = (8,6))

sns.countplot(y = rg['channelGrouping'], order = rg['channelGrouping'].value_counts().index, palette='Blues_d')

plt.xlabel("Count of Revenue generating customers", fontsize = 14)

plt.ylabel("Channel Grouping",fontsize = 14)

plt.title("Channel Grouping")

plt.show()
channel_revenue_group = train.groupby('channelGrouping')['totals_transactionRevenue'].sum().apply(lambda x: x/10000)



plt.figure(figsize = (8,6))

sns.barplot(y = channel_revenue_group.index, x = channel_revenue_group.values,palette='Blues_d', order = channel_revenue_group.sort_values(ascending = False).index)

plt.xlabel("Total Revenue", fontsize = 14)

plt.ylabel("Channel Grouping",fontsize = 14)

plt.title("Total Revenue through various channels")

plt.show()
train.groupby('channelGrouping')['totals_transactionRevenue'].max().apply(lambda x: x/10000).sort_values(ascending = False)
plt.figure(figsize = (8,6))

sns.countplot(y = train['device_browser'], order = train['device_browser'].value_counts().head(8).index, palette='Blues_d')

plt.xlabel("Count", fontsize = 14)

plt.ylabel("Device Browsers",fontsize = 14)

plt.title("Device Browsers")

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(y = rg['device_browser'], order = rg['device_browser'].value_counts().index, palette='Blues_d')

plt.xlabel("Count of revenue generating customers", fontsize = 14)

plt.ylabel("Device Browser",fontsize = 14)

plt.title("Browsers of revenue generating customers")

plt.show()
browser_revenue_group = rg.groupby('device_browser')['totals_transactionRevenue'].sum().apply(lambda x: x/10000)



plt.figure(figsize = (8,6))

sns.barplot(y = browser_revenue_group.index, x = browser_revenue_group.values,palette='Blues_d', order = browser_revenue_group.sort_values(ascending = False).index)

plt.xlabel("Revenue", fontsize = 14)

plt.ylabel("Device Browser",fontsize = 14)

plt.title("Revenues through various browsers")

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(y = rg['device_browser'], order = rg['device_browser'].value_counts().head(5).index, hue = rg['device_isMobile'], palette='Blues_d')

plt.xlabel("Count", fontsize = 14)

plt.ylabel("Device Browser/Device type",fontsize = 14)

plt.title("Count of Device browsers and device types of revenue generating customers")

plt.show()


plt.figure(figsize = (8,6))

sns.countplot(x = train['device_deviceCategory'], order = train.device_deviceCategory.value_counts().index, palette='Blues_d')

plt.xlabel("Device type", fontsize = 14)

plt.ylabel("Count",fontsize = 14)

plt.title("Device categories")

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(x = rg['device_deviceCategory'], order = rg['device_deviceCategory'].value_counts().index, palette='Blues_d')

plt.xlabel("Device Category", fontsize = 14)

plt.ylabel("Count",fontsize = 14)

plt.title("Devices")

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(y = train['device_operatingSystem'], order = train.device_operatingSystem.value_counts().head(8).index, palette='Blues_d')

plt.xlabel("Count", fontsize = 14)

plt.ylabel("OS",fontsize = 14)

plt.title("Operating systems")

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(y = rg['device_operatingSystem'], order = rg['device_operatingSystem'].value_counts().head(8).index, palette='Blues_d')

plt.xlabel("Count of revenue generating customers", fontsize = 14)

plt.ylabel("OS",fontsize = 14)

plt.title("Channel Grouping")

plt.show()
os_revenue_group = train.groupby('device_operatingSystem')['totals_transactionRevenue'].sum().apply(lambda x: x/10000)



plt.figure(figsize = (8,6))

sns.barplot(y = os_revenue_group.index, x = os_revenue_group.values,palette='Blues_d',order = os_revenue_group.sort_values(ascending = False).head(7).index)

plt.xlabel("Total Revenue", fontsize = 14)

plt.ylabel("OS",fontsize = 14)

plt.title("Total revenue by OS")

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(y = rg['device_operatingSystem'], order = rg['device_operatingSystem'].value_counts().head(6).index, hue = rg['device_deviceCategory'], palette='Blues_d')

plt.xlabel("Count", fontsize = 14)

plt.ylabel("OS/Device",fontsize = 14)

plt.title("Customers across where devices/OS")

plt.show()
geo_cols = [col for col in train.columns if 'geoNetwork' in col]
plt.figure(figsize = (8,6))

sns.countplot(y = train['geoNetwork_continent'], order = train['geoNetwork_continent'].value_counts().index)

plt.xlabel("Count", fontsize = 14)

plt.ylabel("Continenet",fontsize = 14)

plt.title("users across various continents")

plt.show()
plt.figure(figsize = (14,10))

sns.countplot(x = train['geoNetwork_country'], order = train['geoNetwork_country'].value_counts().head(20).index)

plt.xticks(rotation=90)

plt.xlabel("Country", fontsize = 14)

plt.ylabel("Count",fontsize = 14)

plt.title("users across various countries")

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(y = rg['geoNetwork_country'], order = rg['geoNetwork_country'].value_counts().head(10).index, palette='Blues_d')

plt.xlabel("Country", fontsize = 14)

plt.ylabel("Count",fontsize = 14)

plt.title("revenue generating customers across various countries")

plt.show()
us_traffic = train[train['geoNetwork_country'] == 'United States']



plt.figure(figsize = (10,8))

sns.countplot(y = us_traffic['geoNetwork_metro'], order = us_traffic['geoNetwork_metro'].value_counts().head(10).index)

plt.xlabel("City", fontsize = 14)

plt.ylabel("Count",fontsize = 14)

plt.title("Customers across various US cities")

plt.show()
tmp = train['geoNetwork_country'].value_counts()

country_visits = pd.DataFrame(data={'geoNetwork_country': tmp.values}, index=tmp.index).reset_index()

country_visits.columns = ['Country', 'Visits']
colorscale = [[0, 'rgb(102,194,165)'], [0.005, 'rgb(102,194,165)'], 

              [0.01, 'rgb(171,221,164)'], [0.02, 'rgb(230,245,152)'], 

              [0.04, 'rgb(255,255,191)'], [0.05, 'rgb(254,224,139)'], 

              [0.10, 'rgb(253,174,97)'], [0.25, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]



def plot_country_map(data, location, z, legend, title, colormap='Rainbow'):

    data = dict(type = 'choropleth', 

                colorscale = colorscale,

                autocolorscale = False,

                reversescale = False,

               locations = data[location],

               locationmode = 'country names',

               z = data[z], 

               text = data[z],

               colorbar = {'title':legend})

    layout = dict(title = title, 

                 geo = dict(showframe = False, 

                         projection = {'type': 'natural earth'}))

    choromap = go.Figure(data = [data], layout=layout)

    iplot(choromap)
plot_country_map(country_visits, 'Country', 'Visits', 'Visits', 'Visits per country')

plt.show()
revenue_by_country = rg.groupby('totals_transactionRevenue')['geoNetwork_country'].sum().reset_index()



plot_country_map(revenue_by_country, 'geoNetwork_country', 'totals_transactionRevenue', 'Transaction Revenue', 'Revenue per country')

plt.show()
train['date'] = pd.to_datetime(train['date'], format = '%Y%m%d')

test['date'] = pd.to_datetime(test['date'], format = '%Y%m%d')
def plot_scatter_data(data, xtitle, ytitle, title, color='blue'):

    trace = go.Scatter(

        x = data.index,

        y = data.values,

        name=ytitle,

        marker=dict(

            color=color,

        ),

        mode='lines+markers'

    )

    data = [trace]

    layout = dict(title = title,

              xaxis = dict(title = xtitle), yaxis = dict(title = ytitle),

             )

    fig = dict(data=data, layout=layout)

    iplot(fig, filename='lines')
count_all = train.groupby('date')['totals_transactionRevenue'].agg(['size'])

count_all.columns = ["Total"]

count_all = count_all.sort_index()

plot_scatter_data(count_all['Total'],'Date', 'Total','Total count of visits (including zero transactions)','green')
count_nonzero = train.groupby('date')['totals_transactionRevenue'].agg(['count'])

count_nonzero.columns = ["Total"]

count_nonzero = count_nonzero.sort_index()

plot_scatter_data(count_nonzero['Total'],'Date', 'Total','Total non-zero transaction visits','darkblue')
total_nonzero = train.groupby('date')['totals_transactionRevenue'].agg(['sum'])

total_nonzero.columns = ["Total"]

total_nonzero = total_nonzero.sort_index()

plot_scatter_data(total_nonzero['Total'],'Date', 'Total','Total non-zero transaction amounts','red')
train_df_date = train.set_index('date')

train_df_date['year'] = train_df_date.index.year

train_df_date['month'] = train_df_date.index.month

train_df_date['weekday'] = train_df_date.index.weekday_name

#train_df_date.head()

train_df_date_rg = train_df_date[train_df_date['totals_transactionRevenue'] > 0]
plt.figure(figsize = (8,6))

sns.countplot(train_df_date['year'], palette = 'Blues_d')

plt.xlabel("Year", fontsize = 14)

plt.ylabel("Count",fontsize = 14)

plt.title("Users over the years")

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(train_df_date_rg['year'], palette = 'Blues_d')

plt.xlabel("Year", fontsize = 14)

plt.ylabel("Count",fontsize = 14)

plt.title("Revenue generating users over the years")

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(train_df_date['month'], palette = 'Blues_d')

plt.xlabel("Month", fontsize = 14)

plt.ylabel("Count",fontsize = 14)

plt.title("Users over the Months")

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(train_df_date_rg['month'], palette = 'Blues_d')

plt.xlabel("Month", fontsize = 14)

plt.ylabel("Count",fontsize = 14)

plt.title("Revenue generating users over the months")

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(train_df_date_rg['weekday'], palette = 'Blues_d', order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

plt.xlabel("Weekday", fontsize = 14)

plt.ylabel("Count",fontsize = 14)

plt.title("Users over week days")

plt.show()
plt.figure(figsize = (8,6))

sns.countplot(train_df_date['weekday'], palette = 'Blues_d', order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

plt.xlabel("Weekday", fontsize = 14)

plt.ylabel("Count",fontsize = 14)

plt.title("Revenue generating users over week days")

plt.show()
from wordcloud import WordCloud



ad_content = train['trafficSource_adContent'].fillna('')

wordcloud_ad = WordCloud(width=800, height=400, background_color="white").generate(' '.join(ad_content))

plt.figure( figsize=(12,9))

plt.imshow(wordcloud_ad)

plt.axis("off")

plt.show()
source = train['trafficSource_source'].fillna('')

wordcloud_source = WordCloud(width=800, height=400, background_color="white").generate(' '.join(source))

plt.figure( figsize=(12,9))

plt.imshow(wordcloud_source)

plt.axis("off")

plt.show()