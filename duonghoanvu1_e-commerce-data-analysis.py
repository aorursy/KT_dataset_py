# Data Processing
import numpy as np
import pandas as pd
import datetime as dt

# Data Visualizing
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from IPython.display import display, HTML
import plotly.express as px
import plotly.graph_objs as go
from IPython.display import display, HTML
from IPython.display import Image

# Data Clustering
from mlxtend.frequent_patterns import apriori # Data pattern exploration
from mlxtend.frequent_patterns import association_rules # Association rules conversion

# Data Modeling
from sklearn.ensemble import RandomForestRegressor

# Math
from scipy import stats  # Computing the t and p values using scipy 
from statsmodels.stats import weightstats 

# Warning Removal
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
# https://stackoverflow.com/questions/22216076/unicodedecodeerror-utf8-codec-cant-decode-byte-0xa5-in-position-0-invalid-s/50538501#50538501
df = pd.read_csv('../input/ecommerce-data/data.csv', encoding= 'unicode_escape')
df
df.describe()
df.info()
df.columns
print(df.duplicated().sum())
df.drop_duplicates(inplace = True)
# https://stackoverflow.com/questions/574730/python-how-to-ignore-an-exception-and-proceed/575711#575711
# https://stackoverflow.com/questions/59127458/pandas-fillna-using-groupby-and-mode
def cleaning_description(df):
    try: 
        return df.mode()[0] # df.mode().iloc[0]
    except Exception:
        return 'unknown'
    
df[['StockCode', 'Description']] = df[['StockCode', 'Description']].fillna(df[['StockCode', 'Description']].groupby('StockCode').transform(cleaning_description))

# Cleaning Description field for proper aggregation
df['Description'] = df['Description'].str.strip().copy()
def clean_InvoiceNo(InvoiceNo):    
    if InvoiceNo[0] == 'C':
        return InvoiceNo.replace(InvoiceNo[0], '')
    else:
        return InvoiceNo
df['InvoiceNo'] = df['InvoiceNo'].apply(clean_InvoiceNo)
# Plot Quantity
plt.figure(constrained_layout=True, figsize=(12, 5))
sns.boxplot(df['Quantity'])

# remove outliers for Quantity
df = df[(df['Quantity'] < 15000) & (df['Quantity'] > -15000)]
# Change datatype of InvoiceDate as datetime type
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
# df['date'] = pd.to_datetime(df['InvoiceDate'], utc=False)
# df['date'].dtypes

# Create new features
df['date'] = df['InvoiceDate'].dt.date   # df['date'].dt.normalize()  # Show only date
df['day'] = df['InvoiceDate'].dt.day
df['month'] = df['InvoiceDate'].dt.month
df['year'] = df['InvoiceDate'].dt.year
df['hour'] = df['InvoiceDate'].dt.hour
df['dayofweek'] = df['InvoiceDate'].dt.dayofweek
df['dayofweek'] = df['dayofweek'].map( {0: '1_Mon', 1: '2_Tue', 2: '3_Wed', 3: '4_Thur', 4: '5_Fri', 5: '6_Sat', 6: '7_Sun'})
# Clean UnitPrice
''' 
Steps to clean Unit Price
    df['UnitPrice'].describe()
    df[df['UnitPrice'] < 0]
    sns.boxplot(df['UnitPrice'])
    sns.distplot(df['UnitPrice'])
    df[df['StockCode'] == 'M']
    df[df['UnitPrice'] > 15000]
'''
df = df[df['UnitPrice'] >= 0]
# Fill CustomerID with unknown
df['CustomerID'].dropna(inplace=True)
# Create a new feature Revenue
df['Revenue'] = df['UnitPrice'] * df['Quantity']
CustomerID_Rev = df.groupby('CustomerID')[['Revenue',
                          'Quantity',
                          'UnitPrice']].agg(['sum',
                                             'mean',
                                             'median']).sort_values(by=[('Revenue', 'sum')], ascending=False)
display(CustomerID_Rev.reset_index())

display(pd.DataFrame(CustomerID_Rev.iloc[1:][('Revenue','sum')].describe()))

# Remove the unknown CustomerID
sns.distplot(CustomerID_Rev.iloc[1:][('Revenue','sum')], kde=False)
Item_retured = df[df['Quantity'] < 0].groupby('CustomerID')[['Revenue',
                                              'Quantity']].agg(['sum']).sort_values(by=[('Quantity', 'sum')], ascending=True).head(10)

sns.barplot(x=Item_retured.index, y=abs(Item_retured[('Quantity','sum')]))
plt.ylabel('A number of Quantity returned')
plt.xticks(rotation=90)
plt.show()

Item_retured
most_prefered_items = df.groupby(['StockCode', 'UnitPrice'])[['Quantity']].sum().sort_values(by=['Quantity'],ascending=False).head(10)

most_prefered_items
most_prefered_items1 = df.groupby(['StockCode'])[['Quantity']].sum().sort_values(by=['Quantity'],ascending=False).head(10)

most_prefered_items2 = df.groupby(['StockCode', 'UnitPrice'])[['Quantity']].sum().sort_values(by=['Quantity'],ascending=False).head(10)

sns.barplot(x=most_prefered_items1.index, y=most_prefered_items1['Quantity'])
plt.ylabel('A number of Quantity returned')
plt.xticks(rotation=90)
plt.show()

display(most_prefered_items1)
display(most_prefered_items2)
least_prefered_items = df.groupby(['StockCode'])[['Quantity']].sum().sort_values(by=['Quantity'],ascending=False)
least_prefered_items = least_prefered_items[least_prefered_items['Quantity']==0]
print('A list of least preferred items: ', len(least_prefered_items))
least_prefered_items
InvoiceNumber_Country = pd.DataFrame(df.groupby(['Country'])['InvoiceNo'].count())

fig = go.Figure(data=go.Choropleth(
                locations=InvoiceNumber_Country.index, # Spatial coordinates
                z = InvoiceNumber_Country['InvoiceNo'].astype(float), # Data to be color-coded
                locationmode = 'country names', # set of locations match entries in `locations`
                colorscale = 'Reds',
                colorbar_title = "Order number",
            ))

fig.update_layout(
    title_text = 'Order number per country',
    geo = dict(showframe = True, projection={'type':'mercator'})
)
fig.layout.template = None
fig.show()
# Source: https://stackoverflow.com/questions/36220829/fine-control-over-the-font-size-in-seaborn-plots-for-academic-papers/36222162#36222162
country_revenue = df.groupby('Country')[['Revenue']].agg(['sum',
                                        'mean',
                                        'median']).sort_values(by=[('Revenue', 'sum')], ascending=False)
display(country_revenue)

fig = plt.figure(constrained_layout=True, figsize=(20, 6))
a = sns.barplot(y=country_revenue.index, x=country_revenue[('Revenue', 'sum')])
plt.xlabel('Total Revenue from all country', fontsize=18)
plt.ylabel('Country', fontsize=18)


fig = plt.figure(constrained_layout=True, figsize=(20, 6))
country_revenue = country_revenue.drop('United Kingdom')
sns.barplot(y=country_revenue.index, x=country_revenue[('Revenue', 'sum')])
plt.xlabel('Total Revenue from all country but UK', fontsize=18)
plt.ylabel('Country', fontsize=18)
plt.show()


country_quantity = df.groupby('Country')[['Quantity']].agg(['sum',
                                        'mean',
                                        'median']).sort_values(by=[('Quantity', 'sum')], ascending=False)

display(country_quantity)

fig = plt.figure(constrained_layout=True, figsize=(20, 6))
a = sns.barplot(y=country_quantity.index, x=country_quantity[('Quantity', 'sum')])
plt.xlabel('Total Quantity from all country', fontsize=18)
plt.ylabel('Country', fontsize=18)


fig = plt.figure(constrained_layout=True, figsize=(20, 6))
country_quantity = country_quantity.drop('United Kingdom')
sns.barplot(y=country_quantity.index, x=country_quantity[('Quantity', 'sum')])
plt.xlabel('Total Quantity from all country but UK', fontsize=18)
plt.ylabel('Country', fontsize=18)
plt.show()
unitprice_average = df.groupby('Country')[['UnitPrice']].agg(['sum',
                                        'mean']).sort_values(by=[('UnitPrice', 'mean')], ascending=False)
display(unitprice_average)

fig = plt.figure(constrained_layout=True, figsize=(20, 6))
a = sns.barplot(y=unitprice_average.index, x=unitprice_average[('UnitPrice', 'mean')])
plt.xlabel('Total Quantity from all country', fontsize=18)
plt.ylabel('Country', fontsize=18)


fig = plt.figure(constrained_layout=True, figsize=(20, 6))
unitprice_average = unitprice_average.drop('United Kingdom')
sns.barplot(y=country_quantity.index, x=unitprice_average[('UnitPrice', 'mean')])
plt.xlabel('Total Quantity from all country but UK', fontsize=18)
plt.ylabel('Country', fontsize=18)
plt.show()

month_sales = df.groupby(['month'])['Revenue'].agg(['sum','mean'])

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
axes = axes.flatten()

sns.barplot(x=month_sales.index, y=month_sales['sum'], ax=axes[0]).set_title("Total Revenue over a year")
plt.ylabel('a')
plt.xticks(rotation=90)

sns.barplot(x=month_sales.index, y=month_sales['mean'], ax=axes[1]).set_title("Average Revenue over a year")
plt.xticks(rotation=90)
plt.show()

month_sales
hour_sales = df.groupby(['hour'])['Revenue'].agg(['sum','mean'])

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
axes = axes.flatten()

sns.barplot(x=hour_sales.index, y=hour_sales['sum'], ax=axes[0]).set_title("Total Revenue in a day")
plt.ylabel('a')
plt.xticks(rotation=90)

sns.barplot(x=hour_sales.index, y=hour_sales['mean'], ax=axes[1]).set_title("Average Revenue per Invoice in a day")
plt.xticks(rotation=90)
plt.show()

hour_sales
dayofweek_sales = df.groupby(['dayofweek'])['Revenue'].agg(['sum','mean',])

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
axes = axes.flatten()

sns.barplot(x=dayofweek_sales.index, y=dayofweek_sales['sum'], ax=axes[0]).set_title("Total Revenue over a week")
plt.ylabel('a')
plt.xticks(rotation=90)

sns.barplot(x=dayofweek_sales.index, y=dayofweek_sales['mean'], ax=axes[1]).set_title("Average Revenue over a week")
plt.xticks(rotation=90)
plt.show()

dayofweek_sales
# Get our date range for our data
print('Date Range: %s to %s' % (df['InvoiceDate'].min(), df['InvoiceDate'].max()))

# We're taking all of the transactions that occurred before December 01, 2011 
df = df[df['InvoiceDate'] < '2011-12-01']
# Get total amount spent per invoice and associate it with CustomerID and Country
invoice_customer_df = df.groupby(by=['InvoiceNo', 'InvoiceDate']).agg({'Revenue': sum,'CustomerID': max,'Country': max,}).reset_index()
invoice_customer_df
# Source: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
# We set our index to our invoice date
# And use Grouper(freq='M') groups data by the index 'InvoiceDate' by Month
# We then group this data by CustomerID and count the number of unique repeat customers for that month (data is the month end date)
# The filter fucntion allows us to subselect data by the rule in our lambda function i.e. those greater than 1 (repeat customers)

monthly_repeat_customers_df = invoice_customer_df.set_index('InvoiceDate').groupby([
              pd.Grouper(freq='M'), 'CustomerID']).filter(lambda x: len(x) > 1).resample('M').nunique()['CustomerID']

monthly_repeat_customers_df
# Number of Unique customers per month
monthly_unique_customers_df = df.set_index('InvoiceDate')['CustomerID'].resample('M').nunique()
monthly_unique_customers_df
# Ratio of Repeat to Unique customers
monthly_repeat_percentage = monthly_repeat_customers_df/monthly_unique_customers_df*100.0
monthly_repeat_percentage
fig = plt.figure(constrained_layout=True, figsize=(20, 6))
grid = gridspec.GridSpec(nrows=1, ncols=1,  figure=fig)

ax = fig.add_subplot(grid[0, 0])

pd.DataFrame(monthly_repeat_customers_df.values).plot(ax=ax, figsize=(12,8))

pd.DataFrame(monthly_unique_customers_df.values).plot(ax=ax,grid=True)

ax.set_xlabel('Date')
ax.set_ylabel('Number of Customers')
ax.set_title('Number of Unique vs. Repeat Customers Over Time')
plt.xticks(range(len(monthly_repeat_customers_df.index)), [x.strftime('%m.%Y') for x in monthly_repeat_customers_df.index], rotation=45)
ax.legend(['Repeat Customers', 'All Customers'])
# Let's investigate the relationship between revenue and repeat customers
monthly_revenue_df = df.set_index('InvoiceDate')['Revenue'].resample('M').sum()

monthly_rev_repeat_customers_df = invoice_customer_df.set_index('InvoiceDate').groupby([
    pd.Grouper(freq='M'), 'CustomerID']).filter(lambda x: len(x) > 1).resample('M').sum()['Revenue']

# Let's get a percentage of the revenue from repeat customers to the overall monthly revenue
monthly_rev_perc_repeat_customers_df = monthly_rev_repeat_customers_df/monthly_revenue_df * 100.0
monthly_rev_perc_repeat_customers_df
fig = plt.figure(constrained_layout=True, figsize=(20, 6))
grid = gridspec.GridSpec(nrows=1, ncols=1,  figure=fig)

ax = fig.add_subplot(grid[0, 0])
pd.DataFrame(monthly_rev_repeat_customers_df.values).plot(ax=ax, figsize=(12,8))

pd.DataFrame(monthly_revenue_df.values).plot(ax=ax,grid=True)

ax.set_xlabel('Date')
ax.set_ylabel('Number of Customers')
ax.set_title('Number of Unique vs. Repeat Customers Over Time')
plt.xticks(range(len(monthly_repeat_customers_df.index)), [x.strftime('%m.%Y') for x in monthly_repeat_customers_df.index], rotation=45)
ax.legend(['Repeat Customers', 'All Customers'])
# Now let's get quantity of each item sold per month
date_item_df = df.set_index('InvoiceDate').groupby([pd.Grouper(freq='M'), 'StockCode'])['Quantity'].sum()
date_item_df.head(15)
# Rank items by the last month's sales
last_month_sorted_df = date_item_df.loc['2011-11-30']
last_month_sorted_df = last_month_sorted_df.reset_index()
last_month_sorted_df.sort_values(by='Quantity', ascending=False).head(10)
# Let's look at the top 5 items sale over a year
date_item_df = df.loc[df['StockCode'].isin(['23084', '84826', '22197', '22086', '85099B'])].set_index('InvoiceDate').groupby([
    pd.Grouper(freq='M'), 'StockCode','Description'])['Quantity'].sum().reset_index()

date_item_df
date_item_df = date_item_df.reset_index()

sns.set(style='whitegrid')
plt.figure(constrained_layout=True, figsize=(12, 5))
sns.lineplot(x=date_item_df['InvoiceDate'], y=date_item_df['Quantity'], hue=date_item_df['StockCode'])
df.groupby(['StockCode', 'Description'])['InvoiceNo'].count().sort_values(ascending = False).head(10)
Num_Canceled_Orders = df[df['Quantity']<0]['InvoiceNo'].nunique()
Total_Orders = df['InvoiceNo'].nunique()
print('Cancellation Rate: {:.2f}%'.format(Num_Canceled_Orders/Total_Orders*100 ))
Monthly_Reorder_Items_Revenue = df.set_index('InvoiceDate').groupby([ pd.Grouper(freq='M'), 'StockCode']).filter(lambda x: len(x) > 1).resample('M').sum()['Revenue']
Monthly_One_Items_Revenue = df.set_index('InvoiceDate').groupby([ pd.Grouper(freq='M'), 'StockCode']).filter(lambda x: len(x) == 1).resample('M').sum()['Revenue']
#Monthly_Revenue = df.groupby(['year','month']).sum()['Revenue']  # Generate the same Result
Monthly_Revenue = df.set_index('InvoiceDate').groupby([pd.Grouper(freq='M')]).sum()['Revenue']
fig = plt.figure(constrained_layout=True, figsize=(20, 6))

ax = fig.add_subplot()
pd.DataFrame(Monthly_Reorder_Items_Revenue.values).plot(ax=ax, figsize=(12,8))
pd.DataFrame(Monthly_Revenue.values).plot(ax=ax,grid=True)
pd.DataFrame(Monthly_One_Items_Revenue.values).plot(ax=ax,grid=True)

ax.set_xlabel('Date')
ax.set_ylabel('Number of Customers')
ax.set_title('Number of Unique vs. Repeat vs Total Items Over Time')
plt.xticks(range(len(monthly_repeat_customers_df.index)), [x.strftime('%m.%Y') for x in monthly_repeat_customers_df.index], rotation=45)
ax.legend(['Repeat Items', 'All Items', 'One Item'])
Sample_df = df[:50]
Sample_df = Sample_df[['InvoiceNo', 'Description']]
Sample_df.set_index('InvoiceNo', inplace=True)
# Note that the quantity bought is not considered, only if the item was present or not in the basket
basket = pd.get_dummies(Sample_df)
basket_sets = pd.pivot_table(basket, index='InvoiceNo', aggfunc='sum')
basket_sets
# Apriori aplication: frequent_itemsets
# Note that min_support parameter was set to a very low value, this is the Spurious limitation, more on conclusion section
frequent_itemsets = apriori(basket_sets, min_support=0.22, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
# Advanced and strategical data frequent set selection
frequent_itemsets[ (frequent_itemsets['length'] > 1) &
                   (frequent_itemsets['support'] >= 0.02)]
# Generating the association_rules: rules
# Selecting the important parameters for analysis
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values('support', ascending=False).head()
# Visualizing the rules distribution color mapped by Lift
plt.figure(figsize=(14, 8))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], alpha=0.9, cmap='YlOrRd');
plt.title('Rules distribution color mapped by lift');
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.colorbar();
# df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format="%m/%d/%Y %H:%M")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['Revenue'] = df['Quantity']*df['UnitPrice']
invoice_ct = df.groupby(by='CustomerID', as_index=False)['InvoiceNo'].count()
invoice_ct.columns = ['CustomerID', 'NumberOrders']
invoice_ct
unitprice = df.groupby(by='CustomerID', as_index=False)['UnitPrice'].mean()
unitprice.columns = ['CustomerID', 'Unitprice']
unitprice
revenue = df.groupby(by='CustomerID', as_index=False)['Revenue'].sum()
revenue.columns = ['CustomerID', 'Revenue']
revenue
total_items = df.groupby(by='CustomerID', as_index=False)['Quantity'].sum()
total_items.columns = ['CustomerID', 'NumberItems']
total_items
earliest_order = df.groupby(by='CustomerID', as_index=False)['InvoiceDate'].min()
earliest_order
earliest_order.columns = ['CustomerID', 'EarliestInvoice']
earliest_order['now'] = pd.to_datetime((df['InvoiceDate']).max())
earliest_order
# == earliest_order['days_as_customer'] = 1 + (earliest_order.now-earliest_order.EarliestInvoice).dt.days
# Source: https://kite.com/python/docs/pandas.core.indexes.accessors.TimedeltaProperties
earliest_order['days_as_customer'] = 1 + (earliest_order['now']-earliest_order['EarliestInvoice']).dt.days
earliest_order.drop('now', axis=1, inplace=True)
earliest_order
# when was their last order and how long ago was that from the last date in file (presumably
# when the data were pulled)
last_order = df.groupby(by='CustomerID', as_index=False)['InvoiceDate'].max()
last_order.columns = ['CustomerID', 'last_purchase']
last_order['now'] = pd.to_datetime((df['InvoiceDate']).max())
last_order['days_since_last_purchase'] = 1 + (last_order.now-last_order.last_purchase).astype('timedelta64[D]')
last_order.drop('now', axis=1, inplace=True)
last_order
#combine all the dataframes into one
import functools
dfs = [invoice_ct,unitprice,revenue,earliest_order,last_order,total_items]
CustomerTable = functools.reduce(lambda left,right: pd.merge(left,right,on='CustomerID', how='outer'), dfs)
CustomerTable['OrderFrequency'] = CustomerTable['NumberOrders']/CustomerTable['days_as_customer']
CustomerTable
CustomerTable.corr()['Revenue'].sort_values(ascending = False)
x = CustomerTable[['NumberOrders','Unitprice', 'days_as_customer', 'days_since_last_purchase', 'NumberItems', 'OrderFrequency']]
y = CustomerTable['Revenue']
reg = RandomForestRegressor()
reg.fit(x.values, y)

#list(zip(x, reg.feature_importances_))
coef = pd.Series(reg.feature_importances_, index = x.columns)

imp_coef = coef.sort_values()
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Linear Model")
recency = df.groupby(by='CustomerID', as_index=False)['InvoiceDate'].max()
recency.columns = ['CustomerID', 'last_purchase']
recency['now'] = pd.to_datetime((df['InvoiceDate']).max())
recency['Recency'] = 1 + (recency.now-recency['last_purchase']).astype('timedelta64[D]')
recency.drop(['now','last_purchase'], axis=1, inplace=True)
recency.head()
#check frequency of customer means how many transaction has been done..

frequency = df.copy()
frequency.drop_duplicates(subset=['CustomerID','InvoiceNo'], keep="first", inplace=True) 
frequency = frequency.groupby('CustomerID',as_index=False)['InvoiceNo'].count()
frequency.columns = ['CustomerID','Frequency']
frequency.head()
monetary=df.groupby('CustomerID',as_index=False)['Revenue'].sum()
monetary.columns = ['CustomerID','Monetary']
monetary.head()
dfs = [recency, frequency, monetary]
rfm = functools.reduce(lambda left,right: pd.merge(left,right,on='CustomerID', how='outer'), dfs)
rfm
#bring all the quartile value in a single dataframe
rfm_segmentation = rfm.copy()
rfm_segmentation
from sklearn.cluster import KMeans
SSE_to_nearest_centroid = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(rfm_segmentation)
    SSE_to_nearest_centroid.append(kmeans.inertia_)

plt.figure(figsize=(20,8))
plt.plot(range(1,15),SSE_to_nearest_centroid,"-o")
plt.title("SSE / K Chart", fontsize=18)
plt.xlabel("Amount of Clusters",fontsize=14)
plt.ylabel("Inertia (Mean Distance)",fontsize=14)
plt.xticks(range(1,20))
plt.grid(True)
plt.show()
#fitting data in Kmeans theorem.
kmeans = KMeans(n_clusters=3, random_state=0).fit(rfm_segmentation)

# this creates a new column called cluster which has cluster number for each row respectively.
rfm_segmentation['cluster'] = kmeans.labels_
rfm_segmentation.head()
plt.figure(figsize=(8,5))
sns.boxplot(rfm_segmentation['cluster'],rfm_segmentation.Recency)

plt.figure(figsize=(8,5))
sns.boxplot(rfm_segmentation['cluster'],rfm_segmentation.Frequency)

plt.figure(figsize=(8,5))
sns.boxplot(rfm_segmentation['cluster'],rfm_segmentation.Frequency)
quantile = rfm.quantile(q=[0.25,0.5,0.75])
quantile
# lower the recency, good for store..
def RScore(x):
    if x <= quantile['Recency'][0.25]:
        return 1
    elif x <= quantile['Recency'][0.50]:
        return 2
    elif x <= quantile['Recency'][0.75]: 
        return 3
    else:
        return 4

# higher value of frequency and monetary lead to a good consumer.
def FScore(x):
    if x <= quantile['Frequency'][0.25]:
        return 4
    elif x <= quantile['Frequency'][0.50]:
        return 3
    elif x <= quantile['Frequency'][0.75]: 
        return 2
    else:
        return 1

def MScore(x):
    if x <= quantile['Monetary'][0.25]:
        return 4
    elif x <= quantile['Monetary'][0.50]:
        return 3
    elif x <= quantile['Monetary'][0.75]: 
        return 2
    else:
        return 1
rfm_segmentation
rfm_segmentation['R_quartile'] = rfm_segmentation['Recency'].apply(RScore)
rfm_segmentation['F_quartile'] = rfm_segmentation['Frequency'].apply(FScore)
rfm_segmentation['M_quartile'] = rfm_segmentation['Monetary'].apply(MScore)
rfm_segmentation
# Approach 1: group customer's attributes, leading to detail customer's profile
# for example 121 and 112 are different.
rfm_segmentation['RFMScore'] = rfm_segmentation['R_quartile'].astype(str) \
                               + rfm_segmentation['F_quartile'].astype(str) \
                               + rfm_segmentation['M_quartile'].astype(str)
# Approach 2: group customer's attributes, leading to more general customers' profile
# for example 121 and 112 are the same.
rfm_segmentation['TotalScore'] = rfm_segmentation['R_quartile'] \
                               + rfm_segmentation['F_quartile'] \
                               + rfm_segmentation['M_quartile']
print("Best Customers: ",len(rfm_segmentation[rfm_segmentation['RFMScore']=='111']))
print('Loyal Customers: ',len(rfm_segmentation[rfm_segmentation['F_quartile']==1]))
print("Big Spenders: ",len(rfm_segmentation[rfm_segmentation['M_quartile']==1]))
print('Almost Lost: ', len(rfm_segmentation[rfm_segmentation['RFMScore']=='134']))
print('Lost Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='344']))
print('Lost Cheap Customers: ',len(rfm_segmentation[rfm_segmentation['RFMScore']=='444']))

Image(url= "https://i.imgur.com/YmItbbm.png?")
rfm_segmentation.sort_values(by=['RFMScore', 'Monetary'], ascending=[True, False])
rfm_segmentation.groupby('RFMScore')['Monetary'].mean()
Score_Recency = rfm_segmentation.groupby('TotalScore')['Recency'].mean().reset_index()
Score_Monetatry = rfm_segmentation.groupby('TotalScore')['Monetary'].mean().reset_index()
Score_Frequency = rfm_segmentation.groupby('TotalScore')['Frequency'].mean().reset_index()
sns.barplot(x=Score_Recency['TotalScore'],y=Score_Recency['Recency'])

plt.figure(constrained_layout=True, figsize=(12, 4))

plt.subplot(1,2,1)
sns.barplot(x=Score_Frequency['TotalScore'],y=Score_Frequency['Frequency'])

plt.subplot(1,2,2)
sns.barplot(x=Score_Monetatry['TotalScore'],y=Score_Monetatry['Monetary'])
plt.subplots_adjust(wspace = 0.2)
def get_month(x): 
    return dt.datetime(x.year, x.month, 1)
df['InvoiceMonth'] = df['InvoiceDate'].apply(get_month)
# https://stackoverflow.com/questions/27517425/apply-vs-transform-on-a-group-object
# explain the difference between   apply - transform. In this case, use transform for CohortMonth.
# CohortMonth: the first time a customer came to our retail store.
df['CohortMonth'] = df.groupby('CustomerID')['InvoiceMonth'].transform('min')
def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day
invoice_year, invoice_month, _ = get_date_int(df, 'InvoiceMonth')
cohort_year, cohort_month, _ = get_date_int(df, 'CohortMonth')

years_diff = invoice_year - cohort_year
months_diff = invoice_month - cohort_month

df['CohortIndex'] = years_diff * 12 + months_diff + 1

df.head()
## grouping customer berdasarkan masing masing cohort
cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
# To solve the problem when ploting heatmap diagram below.
cohort_data['CohortMonth'] = cohort_data['CohortMonth'].dt.date
cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
cohort_counts
cohort_sizes = cohort_counts.iloc[:,0]
retention = cohort_counts.divide(cohort_sizes, axis=0)
retention.round(2) * 100
plt.figure(figsize=(15, 8))
plt.title('Retention rates')
sns.heatmap(data = retention,
            annot = True,
            fmt = '.0%',
            vmin = 0.0, vmax = 0.5,
            cmap = 'BuGn')
plt.show()
