import numpy as np

import pandas as pd

import os

import seaborn as sns

sns.set(style="whitegrid")

import matplotlib.pyplot as plt

from numpy import mean, median



import plotly.figure_factory as ff

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

from plotly import tools

from plotly.subplots import make_subplots

# import plotly.plotly

# import plotly.io as pio

init_notebook_mode(connected=True)





from datashader.utils import lnglat_to_meters as webm

import geoviews as gv

import holoviews as hv

import datashader as ds

from holoviews.operation.datashader import datashade, dynspread, rasterize

from colorcet import  rainbow, fire, bjy , cwr

from holoviews.streams import RangeXY

from bokeh.io import push_notebook, show, output_notebook





#display notebook in full width

# from IPython.core.display import display, HTML

# display(HTML("<style>.container { width:100% !important; }</style>"))



pd.set_option('display.max_columns', 100)

def missing_data(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent= (data.isnull().sum() * 100 / data.isnull().count() ).sort_values(ascending = False)

    df = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

    return df[df['Total'] != 0]



def distinct_values(df, cols = None):

    print('Data Shape: No of Rows {}, No of Columns {} \n'.format(df.shape[0], df.shape[1]))

    if cols is None:

       columns = df.columns 

    else:

        columns = cols

    for col in columns:

        dist_vals = df[col].value_counts().shape[0]

        print('Column:{}, Number of distinct Values:{}'.format(col, dist_vals))

        





def display_duplicates(df, col, n_rows = 5):

    num_dups = df[df.duplicated( subset = col, 

                                keep = 'first')].sort_values(by = col).shape[0]

    print('Number of Duplicates for columns {} :{}'.format(col, num_dups))

    return df[df.duplicated(subset = col, keep = False)].sort_values(by = col).head(n_rows)
data_path = '../input/'



cust_data = pd.read_csv(os.path.join(data_path, 'olist_customers_dataset.csv'),dtype={'customer_zip_code_prefix': str})

geo_data  = pd.read_csv(os.path.join(data_path, 'olist_geolocation_dataset.csv'),dtype={'geolocation_zip_code_prefix': str}) 

order_items_data  = pd.read_csv(os.path.join(data_path, 'olist_order_items_dataset.csv')) 

order_payments_data = pd.read_csv(os.path.join(data_path, 'olist_order_payments_dataset.csv')) 

order_reviews_data = pd.read_csv(os.path.join(data_path, 'olist_order_reviews_dataset.csv')) 

orders_data = pd.read_csv(os.path.join(data_path, 'olist_orders_dataset.csv')) 

products_data = pd.read_csv(os.path.join(data_path, 'olist_products_dataset.csv')) 

sellers_data = pd.read_csv(os.path.join(data_path, 'olist_sellers_dataset.csv'),dtype={'seller_zip_code_prefix': str}) 

product_cat_tran_data = pd.read_csv(os.path.join(data_path, 'product_category_name_translation.csv')) 
print('Customer Data - rows:', cust_data.shape[0], 'columns:', cust_data.shape[1])

print('Geo Location  Data - rows:', geo_data.shape[0], 'columns:',geo_data.shape[1])

print('Order Items Data  - rows:', order_items_data.shape[0], 'columns:', order_items_data.shape[1])

print('Order Payments Data  - rows:', order_payments_data.shape[0], 'columns:', order_payments_data.shape[1])

print('Order Reviews Data  - rows:', order_reviews_data.shape[0], 'columns:', order_reviews_data.shape[1])

print('Order Data  - rows:', orders_data.shape[0], 'columns:', orders_data.shape[1])

print('Products Data  - rows:', products_data.shape[0], 'columns:',products_data.shape[1])

print('Sellers Data  - rows:', sellers_data.shape[0], 'columns:', sellers_data.shape[1])

print('Product Translation Data  - rows:', product_cat_tran_data.shape[0], 'columns:', product_cat_tran_data.shape[1])





cust_data.head()
distinct_values(cust_data)
missing_data(cust_data)
geo_data.head()
distinct_values(geo_data)
display_duplicates(geo_data, geo_data.columns.tolist(), n_rows=5)
missing_data(geo_data)
order_items_data.head()
cat_cols = ['order_id', 'order_item_id','product_id', 'seller_id']

num_cols = ['price', 'freight_value']

distinct_values(order_items_data, cat_cols)
cols = order_items_data.columns.tolist()

cols.remove('order_item_id')

display_duplicates(order_items_data, cols)

order_items_data[num_cols].describe()
missing_data(order_items_data)
order_payments_data.head()
cols = order_payments_data.columns.tolist()

num_cols = ['payment_value']

cat_cols = list(set(cols)- set(num_cols))

distinct_values(order_payments_data,cat_cols)
order_payments_data[num_cols].describe()
display_duplicates(order_payments_data, ['order_id'])
order_reviews_data.head()
cat_cols = ['review_id', 'order_id', 'review_score']

distinct_values(order_reviews_data, cat_cols)


display_duplicates(order_reviews_data, ['review_id'])

display_duplicates(order_reviews_data, ['order_id'])

order_reviews_data['review_score'].describe()
missing_data(order_reviews_data)
orders_data.head()
distinct_values(orders_data)
missing_data(orders_data)
products_data.head()
distinct_values(products_data)
missing_data(products_data)
sellers_data.head()
distinct_values(sellers_data)
missing_data(sellers_data)
product_cat_tran_data.head()
distinct_values(product_cat_tran_data)
missing_data(product_cat_tran_data)


order_reviews_data['neg_review'] = order_reviews_data['review_score'].apply(lambda x: 1 if x < 3 else 0)

order_reviews_data['review_class_name'] = order_reviews_data['review_score'].apply(lambda x: 'positive' if x >=3 else 'negative')
orders_data[['order_purchase_timestamp', 'order_delivered_customer_date']] = orders_data[['order_purchase_timestamp', 'order_delivered_customer_date']].apply(pd.to_datetime)

orders_data['days_delivery'] =  (orders_data['order_delivered_customer_date'] -  orders_data['order_purchase_timestamp']).dt.days

orders_data['order_date'] = pd.to_datetime(orders_data['order_purchase_timestamp']).dt.date

orders_data['order_year'] = pd.to_datetime(orders_data['order_purchase_timestamp']).dt.year

orders_data['order_month'] = pd.to_datetime(orders_data['order_purchase_timestamp']).dt.month

orders_data['order_month_name'] = pd.to_datetime(orders_data['order_purchase_timestamp']).dt.month_name()

# Transfrom the Lattitude and Longitude to Mercetor x, y co-ordnates

x, y = webm(geo_data.geolocation_lng, geo_data.geolocation_lat)

geo_data['x'] = pd.Series(x)

geo_data['y'] = pd.Series(y)



#First 3 digits of zip code to cover a wide area

geo_data['geolocation_zip_code_prefix_3_digits']  = geo_data['geolocation_zip_code_prefix'].str[0:3]

cust_data['customer_zip_code_prefix_3_digits']    =  cust_data['customer_zip_code_prefix'].str[0:3]



# transforming the prefixes to int for plotting purposes

geo_data['geolocation_zip_code_prefix'] = geo_data['geolocation_zip_code_prefix'].astype(int)

geo_data['geolocation_zip_code_prefix_3_digits'] = geo_data['geolocation_zip_code_prefix_3_digits'].astype(int)

cust_data['customer_zip_code_prefix_3_digits'] = cust_data['customer_zip_code_prefix_3_digits'].astype(int)
# Remove Ouliers else map will display data from Europe too

#Brazils most Northern spot is at 5 deg 16′ 27.8″ N latitude.;

geo_data = geo_data[geo_data.geolocation_lat <= 5.27438888]

#it’s most Western spot is at 73 deg, 58′ 58.19″W Long.

geo_data = geo_data[geo_data.geolocation_lng >= -73.98283055]

#It’s most southern spot is at 33 deg, 45′ 04.21″ S Latitude.

geo_data = geo_data[geo_data.geolocation_lat >= -33.75116944]

#It’s most Eastern spot is 34 deg, 47′ 35.33″ W Long.

geo_data = geo_data[geo_data.geolocation_lng <=  -34.79314722]
cols = order_items_data.columns.tolist()

cols.remove('order_item_id')

order_items_data.drop_duplicates(subset = cols, keep = 'first', inplace = True)

order_items_data.shape
display_duplicates(order_items_data, ['order_id', 'price'], 5)
order_items_data.drop_duplicates(subset = ['order_id'] , keep ='first', inplace = True)

order_items_data.shape
order_reviews_data.drop_duplicates(subset = ['order_id'], keep = 'first', inplace = True)

order_reviews_data.shape
data = pd.merge(orders_data, order_reviews_data, on = 'order_id' )

data.shape
data = pd.merge(data, order_items_data, on = 'order_id' )

data.shape
products_data = pd.merge( products_data, product_cat_tran_data, on = 'product_category_name', how = 'left')

data = pd.merge(data, products_data, on = 'product_id' )

data.shape
data = pd.merge(data, cust_data, on = 'customer_id' )

data.shape
data = pd.merge(data, sellers_data, on = 'seller_id' )

data.shape
data.head()
distinct_values(data)
missing_data(data)
data.describe()
data.groupby('review_score').agg([np.mean])
def get_bar_trace_vert(df):

    # Display Ratings Count

    trace =  go.Bar(

                x= df.index,

                y= df.values.ravel(),

                orientation = 'v'  ,

               

                ) 

    return trace



def get_bar_trace_hor(df):

    # Display Ratings Count

    trace =  go.Bar(

                y= df.index,

                x= df.values.ravel(),

                orientation = 'h'  ,

               

                ) 

    return trace







def display_sales_data(df, xlabel, title):

    trace1 = get_bar_trace_vert(df['count'])

    trace2 = get_bar_trace_vert(df['total_sales'])





    fig = make_subplots(rows=1, cols=2, subplot_titles=('Orders Count', 'Total Sales in BRL' ))

    fig['layout']['xaxis1'].update(title= xlabel , type = 'category')

    fig['layout']['xaxis2'].update(title= xlabel , type = 'category')



    fig['layout']['yaxis1'].update( title= 'Count')

    fig['layout']['yaxis2'].update( title= 'Sales Amount' )



    fig['layout'].update(height = 400, width = 1200, showlegend = False, title = title)

    fig.append_trace(trace1, 1,1)

    fig.append_trace(trace2, 1,2)



    iplot(fig)
df = data.groupby('order_year').agg({'order_year':['count'],  'price':['sum']})[['order_year','price']]

df.columns = [ 'count', 'total_sales']

display_sales_data(df, xlabel = 'Year' , title = 'Yearly Sales')
df = data.groupby(['order_month', 'order_month_name','order_year']).agg({'order_month':['count'],  'price':['sum']})

df.columns = [ 'count', 'total_sales']

df = df.reset_index(['order_month', 'order_year'])

df = df[df['order_year'] == 2017]

display_sales_data(df,  xlabel = 'Month' , title = 'Monthly Sales')



df = data.groupby('order_date').agg({'price':'sum'}).reset_index()

df.columns = ['date','sales']

trace =  go.Scatter(x=df.date, y=df['sales'])

plot_data = [trace]





layout = dict(

    title='Daily Sales Amount',

    xaxis=dict(

               rangeslider=dict( visible = True ),

               type='date'

              )

            )



fig = dict(data = plot_data, layout = layout)

iplot(fig)

# df.to_csv(os.path.join(graphs_path, 'daily_sales.csv'), index = False)   
df = data.groupby('review_score')['review_score'].agg(['size'])

df.columns = ['count']

df['percent'] = df['count']/df['count'].sum()





trace1 = get_bar_trace_hor(df['count'])

trace2 = get_bar_trace_hor(df['percent'])





fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Ratings Count', 'Ratings Percent' ))

fig['layout']['xaxis1'].update(title= 'count')

fig['layout']['xaxis2'].update(title=  'percent', tickformat=".2%" )



fig['layout']['yaxis1'].update( title='Ratings' )

fig['layout']['yaxis2'].update( title='Ratings')



fig['layout'].update(height = 400, width = 1200, showlegend = False)

fig.append_trace(trace1, 1,1)

fig.append_trace(trace2, 1,2)



print("Mean Rating for all orders is {:0.2f}".format(data['review_score'].mean()))

print("% of  orders with negative reviews(ratings 1 or 2) is {:0.2%}".format(df['percent'][df.index < 3].sum()))

iplot(fig)

col = 'product_id'

# col = 'product_category_name_english'

df = data.groupby(col).agg({'review_comment_message':['count']})

df.columns = ['review_comments_count']

df = df.sort_values(by = 'review_comments_count', ascending= False)



trace = go.Histogram(x = df['review_comments_count'],histnorm='probability', nbinsx = 50)

fig = tools.make_subplots(rows = 1, cols =1)

fig.append_trace(trace, 1,1)

fig['layout'].update(height = 400, width = 800, showlegend = False, title = 'Product Review Comments Probability Distribution'.format(col))

fig['layout']['xaxis1'].update(title= 'Number of review comments')

fig['layout']['yaxis1'].update(title= 'Distribution')

iplot(fig)


def  plot_discrete_feature(col, top_n = 10):

    

    df = data.groupby(col).agg({col:['count'],  'neg_review':['mean','sum'],'review_score':['mean'],})[[col, 'neg_review', 'review_score']]

    df.columns = ['count',  'neg_review_perc','neg_review_count', 'review_score_mean',]

    df = df.sort_values(by ='count', ascending= False).head(top_n )



    trace1 = get_bar_trace_hor(df['count'])

    trace2 = get_bar_trace_hor(df['neg_review_perc'])

    trace3 = get_bar_trace_hor(df['neg_review_count'])    

    trace4 = get_bar_trace_hor(df['review_score_mean'])





    fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Number of orders', 'Negative Reviews Percent',

                                                              'Negative Reviews Count', 'Mean Ratings'))

    fig['layout']['xaxis1'].update(title= 'Count')

    fig['layout']['xaxis2'].update(title= 'Percent', tickformat=".2%")

    fig['layout']['xaxis3'].update(title= 'Count')

    fig['layout']['xaxis4'].update(title=  'Mean Rating')



    fig['layout']['yaxis1'].update( title=col, type = 'category',autorange="reversed",  automargin =True )

    fig['layout']['yaxis2'].update( title= col, type = 'category',autorange="reversed",automargin =True)

    fig['layout']['yaxis3'].update( title=col, type = 'category',autorange="reversed",automargin =True )

    fig['layout']['yaxis4'].update( title= col, type = 'category',autorange="reversed",automargin =True)



    fig['layout'].update(height = 1000, width = None, showlegend = False, title = 'Plots for {}'.format(col))

    fig.append_trace(trace1, 1,1)

    fig.append_trace(trace2, 1,2)

    fig.append_trace(trace3, 2,1)

    fig.append_trace(trace4, 2,2)

    

    iplot(fig)

  


plot_discrete_feature(col= 'order_status')



plot_discrete_feature(col= 'product_photos_qty')



plot_discrete_feature(col = 'product_category_name_english', top_n = 20)

plot_discrete_feature(col = 'customer_state', top_n = 20)
plot_discrete_feature(col = 'customer_city', top_n = 20)
plot_discrete_feature(col = 'customer_unique_id', top_n = 20)
plot_discrete_feature(col = 'seller_id', top_n = 20)

plot_discrete_feature(col = 'seller_state', top_n = 20)
# col = 'days_delivery'

# df = data[[col, 'review_score', 'neg_review']].copy()

# df.dropna(subset=[col], inplace =True)

# df_neg =  df[df['neg_review'] == 1] 

# df_pos = df[df['neg_review'] == 0] 



# hist_data = [df_neg[col].values, df_pos[col].values]

# group_labels = ['Negative Reviews', 'Positive Reviews']

# fig = ff.create_distplot(hist_data, group_labels)

# fig['layout'].update(height = 600, width = 800)

# iplot(fig)
def display_continous_features(df, col, xlim = None):

    df = df[[col, 'review_score', 'neg_review', 'review_class_name']].copy()

    df.dropna(subset=[col], inplace =True)



    fig = plt.figure(figsize=(20, 20)) 

    #Display Density Plot

    sns.distplot(df[col], color = 'b',  kde = True, ax = plt.subplot(321)  )

    plt.xlim(right = xlim)

    plt.ylabel('Density')





    #Display Density Plot for negatave vs positive reviews

    sns.distplot(df[df['review_class_name'] == 'negative'][col], color = 'r', label = 'Negative(<3 rating)',ax = plt.subplot(322))

    sns.distplot(df[df['review_class_name'] == 'positive'][col], color = 'b', label = 'Positive (>=3 rating)',ax = plt.subplot(322))

    plt.xlim(right = xlim)

    plt.legend(loc = 'best')

    plt.ylabel('Density negative vs positive reviews')



#   Display Box Plot for feature

    sns.boxplot(x = col , data = df, ax = plt.subplot(323))

    plt.xlim(left = 0, right = xlim)

    

#     Display Violin Plot for survived vs died

    sns.violinplot(x = col , y = 'review_class_name', data = df, ax = plt.subplot(324))

    plt.xlim(right = xlim)

   

    

#   Plot average column value for each rating    

    sns.barplot(x="review_class_name", y= col, data = df, estimator=mean, ax = plt.subplot(325))

    plt.ylabel('Mean {}'.format(col))

    plt.show()
col = 'days_delivery'

display_continous_features(data, col)
null_delivery = data[data['days_delivery'].isnull()]

print('The number of orders not yet delivered {}'.format(null_delivery.shape[0]))

print('Mean Rating for orders not delivered {:0.2f} / 5'.format(null_delivery.review_score.mean()))

col = 'price'

display_continous_features(data, col)
col = 'freight_value'

display_continous_features(data, col, xlim = 100)
def display_cumilative_sales(col, col_name):

    agg_values = {col:'count', 'review_score':'mean', 'neg_review':['sum', 'mean'], 'price':'sum', 'days_delivery':'mean' }

    df = data.groupby(col).agg(agg_values)[[col,'review_score', 'neg_review', 'price', 'days_delivery']]

    df.columns = ['count', 'review_score_mean',  'neg_review_count','neg_review_perc', 'total_sales' ,'days_delivery_mean' ]

    df = df.sort_values(by = 'total_sales', ascending= False).reset_index()



    #Cumilative Count

    df['cum_count'] = df.index + 1

    #Cumilative Count Percentage

    df['cum_count_perc'] = df['cum_count'] / df.shape[0]

    # Cumilative Sales

    df['cum_sales'] = df['total_sales'].cumsum(axis = 0) 

    # Cumilative Sales Percentage

    df['cum_sales_perc'] = df['cum_sales']  / df['total_sales'].sum()



   

    

    trace1 =  go.Scatter(x=df['cum_count'], y=df['cum_sales'], name = 'Sales Amount')

    trace2 =  go.Scatter(x=df['cum_count_perc'], y=df['cum_sales_perc'], name = 'Sales Percentage')



    plot_data = [trace1, trace2]



    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Cumilative Sales Amount', 'Cumilative Sales Percent'))



    fig['layout']['xaxis1'].update(title=  '{} count'.format(col_name))

    fig['layout']['xaxis2'].update(title=  '{} Percentage'.format(col_name), tickformat = '0.0%')

    fig['layout']['yaxis1'].update( title= 'Cumilative Sales Amount')

    fig['layout']['yaxis2'].update( title= 'Cumilative Sales Percent',  tickformat = '0.0%')

    fig['layout'].update(title = 'Cumilative Sales for {}'.format(col_name))



    fig.append_trace(trace1, 1,1)

    fig.append_trace(trace2, 1,2)

    iplot(fig)

    

#   Print TOP revenues 

    for top_perc in [1,10,20,30,50]:

        top_perc_count = int(df.shape[0] * top_perc /100)

        df_top_perc = df.head(top_perc_count)



        print('Top {}% ({}) {}  Generate {:.2%} ({:0.2f}M) of revenue'.format( top_perc,

                                                                                top_perc_count, 

                                                                                col_name,

                                                                                df_top_perc['total_sales'].sum()/df['total_sales'].sum(),

                                                                                df_top_perc['total_sales'].sum() / 10**6

                                                                              ))



display_cumilative_sales('seller_id', 'Sellers')

display_cumilative_sales('product_id', 'Products')

display_cumilative_sales('customer_id', 'Customers')

display_cumilative_sales('customer_city', 'City')
# T = 0.05

# PX = 1



# output_notebook()

# hv.extension('bokeh')



# %opts Overlay [width=800 height=600 toolbar='above' xaxis=None yaxis=None]

# %opts QuadMesh [tools=['hover'] colorbar=True] (alpha=0 hover_alpha=0.2)



# def plot_map(data, label, agg_data, agg_name, cmap):

#     url="http://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{Z}/{Y}/{X}.png"

#     geomap = gv.WMTS(url)

#     points = hv.Points(gv.Dataset(data = data, kdims = ['x', 'y'], vdims = [agg_name]))

#     agg = datashade(points, element_type = gv.Image, aggregator= agg_data, cmap =cmap)

#     zip_codes = dynspread(agg, threshold= T, max_px = PX)

#     hover = hv.util.Dynamic(rasterize(points, aggregator= agg_data, width = 50, height = 25, streams = [RangeXY]), operation= hv.QuadMesh)

#     hover =hover.options(cmap = cmap)

#     img = geomap * zip_codes * hover

#     img = img.relabel(label)

#     return img

# brazil_geo = geo_data.set_index('geolocation_zip_code_prefix_3_digits').copy()

# agg_name = 'geolocation_zip_code_prefix'

# agg_data = ds.min(agg_name)

# cmap = rainbow

# label = 'Brazil ZIP Codes'

# # plot_map(data = brazil_geo, label = 'Brazil Zip Codes', agg_data = agg_data, agg_name = agg_name, cmap = rainbow)
# grp = data.groupby('customer_zip_code_prefix_3_digits')['price'].sum().to_frame()

# revenue = brazil_geo.join(grp)

# agg_name = 'revenue'

# revenue[agg_name] = revenue['price']/1000

# plot_map(revenue, 'Orders Revenue (thousands R$)', ds.mean(agg_name), agg_name, cmap = fire)
# grp = data.groupby('customer_zip_code_prefix_3_digits')['review_score'].mean().to_frame()

# review_score = brazil_geo.join(grp)

# agg_name = 'review_score'

# plot_map(review_score , 'Review Score', ds.mean(agg_name), agg_name, cmap = bjy)
# grp = data.groupby('customer_zip_code_prefix_3_digits')['days_delivery'].mean().to_frame()

# delivery_days = brazil_geo.join(grp)

# agg_name = 'days_delivery'

# plot_map(delivery_days , 'Delivery Days', ds.mean(agg_name), agg_name, cmap = cwr)