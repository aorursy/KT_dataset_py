%matplotlib inline
import matplotlib.pyplot as plt

import pandas as pd
import os



print(os.listdir('../input'))
df = pd.read_csv("../input/Online Retail.csv", sep = ',',encoding = "ISO-8859-1", header= 0)

df.head()
df.shape
df.dtypes
# parse date

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format = "%d-%m-%Y %H:%M")

df.head()
# Visualize the distributuion of Quantity using a box plot



ax = df['Quantity'].plot.box(

    showfliers=False, # without outliers

    grid=True,

    figsize=(10, 7)

)



ax.set_ylabel('Order Quantity')

ax.set_title('Quantity Distribution')



plt.suptitle("")

plt.show()
pd.DataFrame(df['Quantity'].describe())
# Filter out all the cancelled orders



df.loc[df['Quantity'] > 0].shape
df = df.loc[df['Quantity'] > 0]
df.shape
# Look into the numbers of orders received over time



monthly_orders_df = df.set_index('InvoiceDate')['InvoiceNo'].resample('M').nunique()
monthly_orders_df
# Visualize this monthly data with a line chart



ax = pd.DataFrame(monthly_orders_df.values).plot(

    grid=True,

    figsize=(10, 7),

    legend=False

)



ax.set_xlabel('date')

ax.set_ylabel('number of orders/invoices')

ax.set_title('Total Number of Orders Over Time')



plt.xticks(

    range(len(monthly_orders_df.index)), 

    [x.strftime('%m.%Y') for x in monthly_orders_df.index], 

    rotation=45

)



plt.show()
# Look at the data in December 2011



invoice_dates = df.loc[

    df['InvoiceDate'] >= '2011-12-01',

    'InvoiceDate'

]
print('Min date: %s\nMax date: %s' % (invoice_dates.min(), invoice_dates.max()))
# Remove data for December, 2011



df = df.loc[df['InvoiceDate'] < '2011-12-01']
df.shape
monthly_orders_df = df.set_index('InvoiceDate')['InvoiceNo'].resample('M').nunique()
monthly_orders_df
ax = pd.DataFrame(monthly_orders_df.values).plot(

    grid=True,

    figsize=(10,7),

    legend=False

)



ax.set_xlabel('date')

ax.set_ylabel('number of orders')

ax.set_title('Total Number of Orders Over Time')



ax.set_ylim([0, max(monthly_orders_df.values)+500])



plt.xticks(

    range(len(monthly_orders_df.index)), 

    [x.strftime('%m.%Y') for x in monthly_orders_df.index], 

    rotation=45

)



plt.show()
# Built the monthly revenue data column



df['Sales'] = df['Quantity'] * df['UnitPrice']
# Get the monthly revenue data



monthly_revenue_df = df.set_index('InvoiceDate')['Sales'].resample('M').sum()
monthly_revenue_df
# Visualize this data with a line plot



ax = pd.DataFrame(monthly_revenue_df.values).plot(

    grid=True,

    figsize=(10,7),

    legend=False

)



ax.set_xlabel('date')

ax.set_ylabel('sales')

ax.set_title('Total Revenue Over Time')



ax.set_ylim([0, max(monthly_revenue_df.values)+100000])



plt.xticks(

    range(len(monthly_revenue_df.index)), 

    [x.strftime('%m.%Y') for x in monthly_revenue_df.index], 

    rotation=45

)



plt.show()
df.head()
# Aggregate data for each order using InvoiceNo and InvoiceDate



invoice_customer_df = df.groupby(

    by=['InvoiceNo', 'InvoiceDate'],

).agg({

    'Sales': sum,

    'CustomerID': max,

    'Country': max,

}).reset_index()
invoice_customer_df.head()
# Aggregate this data per month and 

# compute the number of customers who made more than one purchase in a given month



monthly_repeat_customers_df = invoice_customer_df.set_index('InvoiceDate').groupby([

    pd.Grouper(freq='M'), 'CustomerID'# group the index InvoiceDate by each month and by CustomerID

]).filter(lambda x: len(x) > 1). resample('M').nunique()['CustomerID']# only those customers with more than one order
monthly_repeat_customers_df
monthly_unique_customers_df = df.set_index('InvoiceDate')['CustomerID'].resample('M').nunique()
monthly_unique_customers_df
# Calculate the percentages of repeat customers for each month



monthly_repeat_percentage = monthly_repeat_customers_df / monthly_unique_customers_df * 100.0

monthly_repeat_percentage
# Visualize all this data in a chart



ax = pd.DataFrame(monthly_repeat_customers_df.values).plot(

    figsize=(10,7)

)



pd.DataFrame(monthly_unique_customers_df.values).plot(

    ax=ax,

    grid=True

)





ax2 = pd.DataFrame(monthly_repeat_percentage.values).plot.bar(

    ax=ax,

    grid=True,

    secondary_y=True, # add another y-axis on the rightside of the chart

    color='green',

    alpha=0.2

)



ax.set_xlabel('date')

ax.set_ylabel('number of customers')

ax.set_title('Number of All vs. Repeat Customers Over Time')



ax2.set_ylabel('percentage (%)')



ax.legend(['Repeat Customers', 'All Customers'])

ax2.legend(['Percentage of Repeat'], loc='upper right')



ax.set_ylim([0, monthly_unique_customers_df.values.max()+100])

ax2.set_ylim([0, 100])



plt.xticks(

    range(len(monthly_repeat_customers_df.index)), 

    [x.strftime('%m.%Y') for x in monthly_repeat_customers_df.index], 

    rotation=45

)



plt.show()
monthly_rev_repeat_customers_df = invoice_customer_df.set_index('InvoiceDate').groupby([

    pd.Grouper(freq='M'), 'CustomerID'# group the index InvoiceDate by each month and by CustomerID

]).filter(lambda x: len(x) > 1). resample('M').sum()['Sales']# sum to add all the sales from repeat customers for a given month
monthly_rev_repeat_customers_df
# Calculate the percentages of the monthly revenue generated by the repeat customers



monthly_rev_perc_repeat_customers_df = monthly_rev_repeat_customers_df/monthly_revenue_df * 100.0
monthly_rev_perc_repeat_customers_df
# Visualize this monthly revenue



ax = pd.DataFrame(monthly_revenue_df.values).plot(figsize=(12,9))



pd.DataFrame(monthly_rev_repeat_customers_df.values).plot(

    ax=ax,

    grid=True,

)



ax.set_xlabel('date')

ax.set_ylabel('sales')

ax.set_title('Total Revenue vs. Revenue from Repeat Customers')



ax.legend(['Total Revenue', 'Repeat Customer Revenue'])



ax.set_ylim([0, max(monthly_revenue_df.values)+100000])



ax2 = ax.twinx()



pd.DataFrame(monthly_rev_perc_repeat_customers_df.values).plot(

    ax=ax2,

    kind='bar',

    color='g',

    alpha=0.2

)



ax2.set_ylim([0, max(monthly_rev_perc_repeat_customers_df.values)+30])

ax2.set_ylabel('percentage (%)')

ax2.legend(['Repeat Revenue Percentage'])



ax2.set_xticklabels([

    x.strftime('%m.%Y') for x in monthly_rev_perc_repeat_customers_df.index

])



plt.show()
# Calculate the number  of items sold for each product for each period



date_item_df = pd.DataFrame(

    df.set_index('InvoiceDate').groupby([

        pd.Grouper(freq='M'), 'StockCode'

    ])['Quantity'].sum()

)



date_item_df
# Rank items by the last month sales

# Specifically, see what items were sold the most on November 30, 2011



last_month_sorted_df = date_item_df.loc['2011-11-30'].sort_values(

    by = 'Quantity', ascending=False

).reset_index()



last_month_sorted_df.head(5)
# Aggregate the monthly sales data for these 5 products



date_item_df = pd.DataFrame(

    df.loc[

        df['StockCode'].isin(['23084', '84826', '22197', '22086', '85099B'])

    ].set_index('InvoiceDate').groupby([

        pd.Grouper(freq='M'), 'StockCode'

    ])['Quantity'].sum()

)



date_item_df
# Transform this data into a tabular format



trending_items_df = date_item_df.reset_index().pivot('InvoiceDate','StockCode').fillna(0)

trending_items_df.head(2)
trending_items_df = trending_items_df.reset_index()

trending_items_df.head(2)
trending_items_df = trending_items_df.set_index('InvoiceDate')

trending_items_df.head(2)
trending_items_df.columns = trending_items_df.columns.droplevel(0)

trending_items_df
# Visualize this time series for the top five best-sellers



ax = pd.DataFrame(trending_items_df.values).plot(

    figsize=(10,7),

    grid=True,

)



ax.set_ylabel('number of purchases')

ax.set_xlabel('date')

ax.set_title('Item Trends over Time')



ax.legend(trending_items_df.columns, loc='upper left')



plt.xticks(

    range(len(trending_items_df.index)), 

    [x.strftime('%m.%Y') for x in trending_items_df.index], 

    rotation=45

)



plt.show()