!pip install plotly_express
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import os

import json

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as px

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def load_df(csv_path, nrows=None):

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    

    df = pd.read_csv(csv_path, 

                     converters={column: json.loads for column in JSON_COLUMNS}, 

                     dtype={'fullVisitorId': 'str'}, # Important!!

                     nrows=nrows)

    

    for column in JSON_COLUMNS:

        column_as_df = pd.json_normalize(df[column])

        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]

        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")

    return df
train_df = load_df('/kaggle/input/ga-customer-revenue-prediction/train.csv')

test_df = load_df('/kaggle/input/ga-customer-revenue-prediction/test.csv')

train_df.to_csv('./train.csv', index=False)

test_df.to_csv('./test.csv', index=False)
train_df = pd.read_csv('./train.csv', low_memory=False)

test_df = pd.read_csv('./test.csv', low_memory=False)
train_df.head()
train_df.info()
train_df['device.isMobile'] = train_df['device.isMobile'].astype(int)
train_df[['totals.visits', 'totals.hits', 'totals.pageviews', 'totals.bounces', 'totals.newVisits', 'totals.transactionRevenue']] = train_df[['totals.visits', 'totals.hits', 'totals.pageviews', 'totals.bounces', 'totals.newVisits', 'totals.transactionRevenue']].astype(float)
# Target Variable

train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].fillna(0)
train_df.describe()
null_df = pd.DataFrame(train_df.isnull().sum()*100/train_df.shape[0], columns=['null_pct']).sort_values(by='null_pct', ascending=True)

null_df = null_df[null_df['null_pct'] > 0]

ax = null_df.plot(kind='barh', figsize=(10,7), color="coral", fontsize=13, legend=False)

for i in ax.patches:

    ax.text(i.get_width()+.3, i.get_y()+0.1, str(round(i.get_width(), 2))+'%', fontsize=15, color='dimgrey')
null_df.index
# Dropping Mostly-Null Columns 

train_df = train_df.drop([

    'sessionId', 'visitId', \

    'trafficSource.campaignCode', 'trafficSource.adContent', 'trafficSource.adwordsClickInfo.gclId', \

    'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.slot',\

    'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.isVideoAd' \

], axis=1)

train_df.shape
cat_df = pd.DataFrame(train_df.apply(lambda x: x.nunique()).sort_values(), columns=['num_uniques'])

na_cols = cat_df[cat_df['num_uniques'] == 1].index

cat_df = cat_df[cat_df['num_uniques'] != 1]

train_df = train_df.drop(na_cols, axis=1)

cat_df
train_df.shape
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')

gdf = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

fig.suptitle('Target Variable - Original vs Log')

ax1.scatter(range(gdf.shape[0]), np.sort(gdf["totals.transactionRevenue"].values))

ax2.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"].values)))
print(f"Percentage of Customers who brought Revenue: {round(100 * sum(gdf['totals.transactionRevenue'] > 0) / gdf['totals.transactionRevenue'].shape[0], 2)}%")
print("Number of unique visitors in train set : ",train_df.fullVisitorId.nunique(), "/", train_df.shape[0])

print("Number of unique visitors in test set : ",test_df.fullVisitorId.nunique(), "/", test_df.shape[0])

print("Number of common visitors in train and test set : ",len(set(train_df.fullVisitorId.unique()).intersection(set(test_df.fullVisitorId.unique())) ))

train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].replace({0: np.nan})
def horizontal_bar_chart(cnt_srs, color):

    trace = go.Bar(

        y=cnt_srs.index[::-1],

        x=cnt_srs.values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace
# Device Browser

cnt_srs = train_df.groupby('device.browser')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')

trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(50, 171, 96, 0.6)')

trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.04, 

                          subplot_titles=["Count", "Non-zero Revenue Count", "Mean Revenue"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)



fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Browser")

py.iplot(fig)

# Device Category

cnt_srs = train_df.groupby('device.deviceCategory')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(71, 58, 131, 0.8)')

trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(71, 58, 131, 0.8)')

trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(71, 58, 131, 0.8)')

# Creating two subplots

fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.04, 

                          subplot_titles=["Count", "Non-zero Revenue Count", "Mean Revenue"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)



fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Category")

py.iplot(fig)

# Operating system

cnt_srs = train_df.groupby('device.operatingSystem')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(246, 78, 139, 0.6)')

trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10),'rgba(246, 78, 139, 0.6)')

trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10),'rgba(246, 78, 139, 0.6)')

# Creating two subplots

fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.04, 

                          subplot_titles=["Count", "Non-zero Revenue Count", "Mean Revenue"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)



fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Operating System")

py.iplot(fig)

# Continent

cnt_srs = train_df.groupby('geoNetwork.continent')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(58, 71, 80, 0.6)')

trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(58, 71, 80, 0.6)')

trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(58, 71, 80, 0.6)')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15, 

                          subplot_titles=["Count", "Non-zero Revenue Count", "Mean Revenue"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)



fig['layout'].update(height=1500, width=1200, paper_bgcolor='rgb(233,233,233)', title="Continent")

py.iplot(fig)

# Sub Continent

cnt_srs = train_df.groupby('geoNetwork.subContinent')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"], 'orange')

trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"], 'orange')

trace3 = horizontal_bar_chart(cnt_srs["mean"], 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15, 

                          subplot_titles=["Count", "Non-zero Revenue Count", "Mean Revenue"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)



fig['layout'].update(height=1500, width=1200, paper_bgcolor='rgb(233,233,233)', title="Sub Continent")

py.iplot(fig)

# Source

cnt_srs = train_df.groupby('trafficSource.source')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'green')

trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'green')

trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'green')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15, 

                          subplot_titles=["Count", "Non-zero Revenue Count", "Mean Revenue"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)



fig['layout'].update(height=1500, width=1200, paper_bgcolor='rgb(233,233,233)', title="Traffic Source")

py.iplot(fig)
# Medium

cnt_srs = train_df.groupby('trafficSource.medium')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])

cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]

cnt_srs = cnt_srs.sort_values(by="count", ascending=False)

trace1 = horizontal_bar_chart(cnt_srs["count"], 'purple')

trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"], 'purple')

trace3 = horizontal_bar_chart(cnt_srs["mean"], 'purple')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15, 

                          subplot_titles=["Count", "Non-zero Revenue Count", "Mean Revenue"])



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)



fig['layout'].update(height=1500, width=1200, paper_bgcolor='rgb(233,233,233)', title="Traffic Medium")

py.iplot(fig)
gdf = train_df.groupby("fullVisitorId").agg({'totals.hits': 'sum', 'totals.pageviews': 'sum', 'totals.transactionRevenue': 'sum'})

gdf['isPayingVisitor'] = gdf['totals.transactionRevenue'] > 0
px.scatter(gdf, x='totals.hits', facet_col='isPayingVisitor')
px.scatter(gdf, x='totals.pageviews', facet_col='isPayingVisitor')