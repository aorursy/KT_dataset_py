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
# import libraries

import bq_helper

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import set_matplotlib_formats

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px



set_matplotlib_formats('retina')

%matplotlib inline



from google.cloud import bigquery

client = bigquery.Client()
# high level stats 



query = """

    SELECT 

        FORMAT("%'d",COUNT(DISTINCT fullVisitorId)) AS users,

        FORMAT("%'d",SUM(totals.visits)) AS visits,

        FORMAT("%'d",SUM(totals.pageviews)) AS pageviews,

        FORMAT("%'d", SUM(totals.transactions)) AS transactions,

        SUM(totals.transactionRevenue)/1000000 AS revenue



    FROM 

        `bigquery-public-data.google_analytics_sample.ga_sessions_*`

    WHERE

        _TABLE_SUFFIX BETWEEN '20161001' AND '20161230'

        AND totals.totalTransactionRevenue IS NOT NULL

        

"""

safe_query_job = client.query(query)

high_level_aug = safe_query_job.to_dataframe()

high_level_aug
# traffic by month

query = """

    SELECT 

        DATE_TRUNC(PARSE_DATE('%Y%m%d',date), MONTH) AS month,

        SUM(totals.visits) AS visits,        

        SUM(totals.transactionRevenue)/1000000 AS revenue



    FROM 

        `bigquery-public-data.google_analytics_sample.ga_sessions_*`

    WHERE

        _TABLE_SUFFIX BETWEEN '20161001' AND '20161230'

        AND totals.totalTransactionRevenue IS NOT NULL

    GROUP BY 1

    ORDER BY 1

        

"""

safe_query_job = client.query(query)

df_1 = safe_query_job.to_dataframe()

df_1.head(3)
# plot web traffic and revenue by month for Q4



fig, ax1 = plt.subplots()



color = 'tab:red'

ax1.set_xlabel('Month')

ax1.set_ylabel('Revenue', color=color)

ax1.plot(df_1['month'], df_1['revenue'], color=color)

ax1.tick_params(axis='y', labelcolor=color)



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



color = 'tab:blue'

ax2.set_ylabel('Visits', color=color)  # we already handled the x-label with ax1

ax2.plot(df_1['month'], df_1['visits'], color=color)

ax2.tick_params(axis='y', labelcolor=color)



plt.title('Monthly Web Traffic and Revenue for 2016Q4', fontsize=14)

plt.xticks(df_1['month'],rotation=45)

plt.show()
# create a table with revenue by country



# traffic by month

query = """

    SELECT 

        geoNetwork.country AS country,   

        SUM(totals.transactionRevenue)/1000000 AS revenue



    FROM 

        `bigquery-public-data.google_analytics_sample.ga_sessions_*`

    WHERE

        _TABLE_SUFFIX BETWEEN '20161001' AND '20161230'

        AND totals.totalTransactionRevenue IS NOT NULL

    GROUP BY 1

    ORDER BY 2 desc

    

"""

safe_query_job = client.query(query)

df_2 = safe_query_job.to_dataframe()

df_2.head(5)
# create a heatmap of revenue by geo location



fig = go.Figure(data=go.Choropleth(

    locations=df_2['country'], # Spatial coordinates

    z = df_2['revenue'].astype(float), # Data to be color-coded

    locationmode = 'country names', # set of locations match entries in `locations`

    colorscale = 'Reds',

    colorbar_title = "revenue USD",

))



fig.update_layout(

    title_text = '2016Q4 Google Merchandise Store by Geo Location',

)



fig.show()
# create a table of metrics by channel



query = """

    SELECT

        channelGrouping as channel,

        SUM(totals.totalTransactionRevenue)/1000000 AS revenue,

        SUM(totals.transactions) AS transactions,

        COUNT(DISTINCT fullVisitorId) AS users,

        SUM(totals.visits) AS sessions,

        SUM(totals.pageviews) AS pageviews

    FROM 

        `bigquery-public-data.google_analytics_sample.ga_sessions_*`

    WHERE

        _TABLE_SUFFIX BETWEEN '20161001' AND '20161230'

        AND totals.totalTransactionRevenue IS NOT NULL

    GROUP BY

        1

    ORDER BY

        2 DESC

"""

safe_query_job = client.query(query)

df_3 = safe_query_job.to_dataframe()

df_3.head(3)
# set up the matplotlib figure

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(10,24))

fig.subplots_adjust(hspace=1)



sns.barplot(x='channel',

            y='revenue',

            data=df_3,

            estimator=sum,

            ax=ax1)



sns.barplot(x='channel',

            y='transactions',

            data=df_3,

            estimator=sum,

            ax=ax2)



sns.barplot(x='channel',

            y='users',

            data=df_3,

            estimator=sum,

            ax=ax3)



sns.barplot(x='channel',

            y='sessions',

            data=df_3,

            estimator=sum,

            ax=ax4)



sns.barplot(x='channel',

            y='pageviews',

            data=df_3,

            estimator=sum,

            ax=ax5)



ax1.set_title('Total Revenue by Channel')

ax2.set_title('Total Transactions by Channel')

ax3.set_title('Total Users by Channel')

ax4.set_title('Total Visits by Channel')

ax5.set_title('Total Pageviews by Channel')
# create a data frame with features

query = """

    SELECT 

        fullVisitorId AS userID,

        AVG(totals.timeOnSite) As avgTimeOnSite,

        SUM(totals.pageviews) AS pageviews,

        SUM(totals.transactions) AS transactions

    FROM 

        `bigquery-public-data.google_analytics_sample.ga_sessions_*`

    WHERE

        _TABLE_SUFFIX BETWEEN '20161001' AND '20161230'

    GROUP BY

        1

    ORDER BY 1

        

"""

safe_query_job = client.query(query)

df_4 = safe_query_job.to_dataframe()

df_4.head(4)
# replace na values with 0

df_4['transactions'].fillna(0, inplace=True)

df_4.head(4)
pearsoncorr = df_4.corr(method='pearson')

pearsoncorr
# create a heat map of features

sns.heatmap(pearsoncorr,

              xticklabels=pearsoncorr.columns,

              yticklabels=pearsoncorr.columns,

              cmap='RdBu_r',

              annot=True,

              linewidth=0.5)
# create a data frame with features

query = """

    SELECT

        hits.page.pagePathLevel1 AS pagePath,

        SUM(totals.pageviews) AS pageviews

    FROM

      `bigquery-public-data.google_analytics_sample.ga_sessions_*`,

      UNNEST(hits) AS hits

    WHERE

        _TABLE_SUFFIX BETWEEN '20161001' AND '20161230'

    GROUP BY 1

    ORDER BY 2 DESC

    LIMIT 10

"""

safe_query_job = client.query(query)

df_5 = safe_query_job.to_dataframe()

df_5
# create a bar chart of top 10 pages with most pageviews

df_5.plot.bar(x='pagePath', y='pageviews', rot=70, title='Top 10 Viewed Pages');

plt.show();
# create a data frame calling product features

query = """

    SELECT

        product.v2ProductCategory AS product_category,

        product.v2ProductName AS product_name,

        product.productSKU AS product_sku,

        product.productPrice/1e6 AS product_price,

        product.productQuantity AS product_quantity,

        product.productRevenue/1e6 AS product_revenue,

        totals.totalTransactionRevenue/1e6 AS total_revenue

    FROM

      `bigquery-public-data.google_analytics_sample.ga_sessions_*`,

      UNNEST(hits) AS hits,

      UNNEST(hits.product) AS product

    WHERE

        _TABLE_SUFFIX BETWEEN '20161001' AND '20161230'

        AND productRevenue IS NOT NULL

"""

safe_query_job = client.query(query)

df_6 = safe_query_job.to_dataframe()

df_6.head(3)
# clean values under product_category

df_6['product_category'].unique()

df_6 = df_6.replace(['${productitem.product.origCatName}'], 'Miscellaneous')
# create a pivot table that shows the Top 10 selling product categories

pivot = pd.pivot_table(df_6, index=['product_category'], values=['product_revenue', 'product_quantity'], aggfunc=np.sum).sort_values(by='product_revenue', ascending=False)

pivot.plot(kind='bar')
# describe product data frame

df_6.describe()
# histogram of product price

hist = sns.distplot(df_6['product_price'], bins=40)

hist.set_title('distribution of product quantity by prices');

hist.set_xlabel('product price');

hist.set_ylabel('frequency');
# distribution of selling price by product category

box_plot = df_6.boxplot(column='product_price', by='product_category', figsize=(16,9));

box_plot.set_title('Product Price by Product Category');

box_plot.set_xlabel('product category');

box_plot.set_ylabel('product price');

box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=45);
# create a data frame calling action features

query = """

    SELECT

        hits.eCommerceAction.action_type AS actions,

        COUNT(fullVisitorId) AS total_hits

    FROM

      `bigquery-public-data.google_analytics_sample.ga_sessions_*`,

      UNNEST(hits) AS hits,

      UNNEST(hits.product) AS product

    WHERE

        _TABLE_SUFFIX BETWEEN '20161001' AND '20161230'

    AND

        (hits.ecommerceaction.action_type != '0' AND hits.ecommerceaction.action_type != '4' AND hits.ecommerceaction.action_type != '3')

    GROUP BY 

        1

    ORDER BY 1

"""

safe_query_job = client.query(query)

df_7 = safe_query_job.to_dataframe()

df_7.head()
# define action types. Action types can be found in BigQuery Export schema: https://support.google.com/analytics/answer/3437719?hl=en

df_7['actions'] = df_7['actions'].astype(str) # convert action types to string



df_7['actions'] = df_7['actions'].replace(['0','1','2','3','4','5','6'],['Unknown', 

                                                                     'Click through of product lists', 

                                                                     'Product detail views', 

                                                                     'Add product(s) to cart',

                                                                     'Remove products from cart',

                                                                     'Check out',

                                                                     'Completed purchase'])

df_7

# create a funnel visualization

fig = go.Figure(go.Funnel(

    y = df_7['actions'],

    x = df_7['total_hits'],

    textposition = 'inside',

    textinfo = 'value+percent initial'))



fig.update_layout(title_text = 'Shopping Cart Abandonment')



fig.show()