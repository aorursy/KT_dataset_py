## Loading the required libraries:
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns
import datetime as dt
import calendar
from scipy.stats import skew,kurtosis

import warnings
warnings.filterwarnings('ignore')

from subprocess import check_output
print(check_output(["ls","../input/olist-v5-data"]).decode("utf8"))
## Reading the datas:
order = pd.read_csv("../input/olist-v5-data/olist_public_dataset_v2.csv")  ## Unclassified orders dataset
customer=pd.read_csv("../input/olist-v5-data/olist_public_dataset_v2_customers.csv")  ### Unique customer id 
payment =pd.read_csv("../input/olist-v5-data/payments_olist_public_dataset.csv")  ### Payment dataset
product = pd.read_csv("../input/brazilian-ecommerce/product_category_name_translation.csv")  ## Product translation to english
geo=pd.read_csv("../input/olist-v5-data/geolocation_olist_public_dataset.csv")  ## Location data
sellers=pd.read_csv("../input/olist-v5-data/sellers_olist_public_dataset_.csv") ## Seller information
order.shape
payment.shape
customer.shape
product.shape
geo.shape
## Joining the order and payment :
#order_pay=pd.merge(order,payment,how="left",on=['order_id','order_id'])
## Joining the order_payment with product category translation :
#order_product=pd.merge(order_pay,product,how="left",on=['product_category_name','product_category_name'])
#Now that we have joined the relevant tables,lets take a look at the data:
print("Total number of orders in the database:",order['order_id'].nunique())
print("Total Number of customers:",order['customer_id'].nunique())
status=order.groupby('order_status')['order_id'].nunique().sort_values(ascending=False)
status
## Executive Summary:
print("Maximum order amount is BRL:",order['order_products_value'].max())
print("Minumum order amount is BRL:",order['order_products_value'].min())
print("Average order value is BRL:",order['order_products_value'].mean())
print("Median order value is BRL:",order['order_products_value'].median())
value = order.groupby('order_id')['order_products_value','order_freight_value'].sum().sort_values(by='order_products_value',ascending=False).reset_index()
value.head()
plt.figure(figsize=(12,10))

plt.subplot(221)
g = sns.distplot(np.log(order['order_products_value'] + 1))
g.set_title("Product Value of Orders - Distribution", fontsize=15)
g.set_xlabel("")
g.set_ylabel("Frequency", fontsize=12)

plt.subplot(222)
g1 = sns.distplot(np.log(order['order_freight_value'] + 1))
g1.set_title("Freight Value of Orders - Distribution", fontsize=15)
g1.set_xlabel("")
g1.set_ylabel("Frequency", fontsize=12)

print("Skewness of the transaction value:",skew(np.log(order['order_products_value']+1)))
print("Excess Kurtosis of the transaction value:",kurtosis(np.log(order['order_products_value']+1)))
order_usual=order.groupby('order_id')['order_items_qty'].aggregate('sum').reset_index()
order_usual=order_usual['order_items_qty'].value_counts()
order_usual.head()
plt.figure(figsize=(8,8))
ax=sns.barplot(x=order_usual.index,y=order_usual.values,color="green")
ax.set_xlabel("Number of products added in order")
ax.set_ylabel("Number of orders")
ax.set_title("Number of products people usually order")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
order_product=pd.merge(order,product,on='product_category_name',how='left')
order_product.shape
most_product=order_product.groupby('product_category_name_english').aggregate({'order_id':'count'}).rename(columns={'order_id':'order_count'}).sort_values(by='order_count',ascending=False).reset_index()
most_product.head()
### Visualising top 10 most bought product categories:
sns.barplot(x='product_category_name_english',y='order_count',data=most_product[:10],color="blue")
plt.xlabel("Product Category")
plt.ylabel("Total Number of orders")
plt.title("Most bought product categories")
plt.xticks(rotation='vertical')
plt.show()
order['order_purchase_timestamp']=pd.to_datetime(order['order_purchase_timestamp'])
order['order_delivered_customer_date']=pd.to_datetime(order['order_delivered_customer_date'])
## Create new columns for date,day,time,month:
order['weekday']=order['order_purchase_timestamp'].dt.weekday_name
order['year']=order['order_purchase_timestamp'].dt.year
order['monthday']=order['order_purchase_timestamp'].dt.day
order['weekday'] = order['order_purchase_timestamp'].dt.weekday
order['month']=order['order_purchase_timestamp'].dt.month
order['hour']=order['order_purchase_timestamp'].dt.hour
# Trend by Year:
trend_year=pd.DataFrame(order.groupby('year')['order_products_value'].sum().sort_values(ascending=False)).reset_index()
ax=sns.barplot(x='year',y='order_products_value',data=trend_year,palette=sns.set_palette(palette='viridis_r'))
#ax.ticklabel_format()
ax.set_xlabel('Year')
ax.set_ylabel('Total Transaction Value')
ax.set_title('Transaction Value by Year')
## Boxplot for transactions by year:
plt.figure(figsize=(8,8))
ax=sns.boxplot(x='year',y='order_products_value',data=order,palette=sns.set_palette(palette='viridis_r'))
ax.set_xlabel('Year')
ax.set_ylabel('Total Value')
ax.set_title('Box Plot of transactions over the year')
## The below code is inspired from Sbans kernel -https://www.kaggle.com/shivamb/deep-exploration-of-gun-violence-in-us 
trend_month=pd.DataFrame(order.groupby('month').agg({'order_products_value':'mean'}).rename(columns={'order_products_value':'mean_transaction'})).reset_index()
x1 = trend_month.month.tolist()
y1 = trend_month.mean_transaction.tolist()
mapp = {}
for m,v in zip(x1, y1):
    mapp[m] = v
xn = [calendar.month_abbr[int(x)] for x in sorted(x1)]
vn = [mapp[x] for x in sorted(x1)]

plt.figure(figsize=(10,7))
ax=sns.barplot(x=xn,y=vn, color='#ed5569')
ax.set_title("Average value of transaction per month")
ax.set_xlabel('Month')
ax.set_ylabel('Value')
trend_weekday=pd.DataFrame(order.groupby('weekday').agg({'order_products_value':'mean'}).rename(columns={'order_products_value':'Mean_Transaction'})).reset_index()
x2 = trend_weekday.index.tolist()
y2 = trend_weekday.Mean_Transaction.tolist()

weekmap = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
x2 = [weekmap[x] for x in x2]
wkmp = {}
for j,x in enumerate(x2):
    wkmp[x] = y2[j]
order_week = list(weekmap.values())
ordervals = [wkmp[val] for val in order_week]

plt.figure(figsize=(10,7))
ax=sns.barplot(x=order_week,y=ordervals, color='#ed5569')
ax.set_title("Average value of transaction by day of the week")
ax.set_xlabel('Day')
ax.set_ylabel('Value')
freq_weekday=pd.DataFrame(order.groupby('weekday').agg({'order_id':'count'}).rename(columns={'order_id':'order_count'})).reset_index()
x3 = freq_weekday.index.tolist()
y3 = freq_weekday.order_count.tolist()

weekmap = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
x3 = [weekmap[x] for x in x3]
wkmp = {}
for j,x in enumerate(x3):
    wkmp[x] = y3[j]
order_week = list(weekmap.values())
ordervals = [wkmp[val] for val in order_week]

plt.figure(figsize=(10,7))
ax=sns.barplot(x=order_week,y=ordervals, palette=sns.color_palette(palette="Set2"))
ax.set_title("Total Number of orders by day of the week")
ax.set_xlabel('Day')
ax.set_ylabel('Value')
week=pd.merge(trend_weekday,freq_weekday,on='weekday',how='inner')
plt.figure(figsize=(8,8))
sns.jointplot(x='Mean_Transaction', y='order_count',data=week, size=10,color='red')
plt.ylabel('Order Count', fontsize=12)
plt.xlabel('Average value of transaction', fontsize=12)
plt.show()
trend_hour=order.groupby('hour').agg({'order_id':'count'}).rename(columns={'order_id':'freq_order'}).reset_index()
plt.figure(figsize=(8,8))
ax=sns.barplot(x=trend_hour['hour'],y=trend_hour['freq_order'],color="red")
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Order Count')
ax.set_title("Frequency of transaction over the hour")
day_hour=order.groupby(['weekday','hour']).agg({'order_id':'count'}).rename(columns={'order_id':'freq'}).reset_index()
day_hour.weekday=day_hour.weekday.map(weekmap)
day_hour.head()
### Sorting it so that the plot order is correct.
day_hour['weekday']=pd.Categorical(day_hour['weekday'],categories=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'],ordered=True)
day_hour=day_hour.pivot('weekday','hour','freq')
plt.figure(figsize=(15,8))
ax=sns.heatmap(day_hour,annot=True,fmt="d",cmap="OrRd")
ax.set_xlabel("Hour")
ax.set_ylabel("Day")
ax.set_title("Heatmap of tranactions over the hour by day",size=10)
trans_state=pd.DataFrame(order.groupby('customer_state').agg({'order_products_value':'mean'}).rename(columns={'order_products_value':'avg_trans'}).sort_values(by='avg_trans',ascending=False)).reset_index()
plt.figure(figsize=(10,7))
ax=sns.barplot(x='customer_state',y='avg_trans',data=trans_state,palette=sns.color_palette(palette="viridis_r"))
ax.set_xlabel('Customer State')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_ylabel('Avg transaction value')
ax.set_title("Average Transaction Value for each state")
### By City :
trans_city=pd.DataFrame(order.groupby('customer_city').agg({'order_products_value':'mean'}).rename(columns={'order_products_value':'avg_trans'}).sort_values(by='avg_trans',ascending=False)).reset_index()
trans_city[:10]
plt.figure(figsize=(10,7))
ax=sns.barplot(x='customer_city',y='avg_trans',data=trans_city[:10],palette=sns.color_palette(palette="Set2"))
ax.set_xlabel('Customer City')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_ylabel('Avg transaction value')
ax.set_title("Top 10 - Average Transaction Value for each City")
order['day_to_delivery']=(order['order_delivered_customer_date']-order['order_purchase_timestamp']).dt.days
print("Average days to delivery {}".format(np.round(order['day_to_delivery'].mean(),0)))
delivery=order.groupby('day_to_delivery')['order_id'].aggregate({'order_id':'count'}).rename(columns={'order_id':'freq'}).reset_index().dropna()
delivery['freq']=delivery['freq'].astype(int)
plt.figure(figsize=(20,10))
sns.barplot(x='day_to_delivery',y='freq',data=delivery,color="blue")
plt.title("Days to delivery")
plt.xlabel("Days")
plt.xticks(rotation="vertical")
plt.ylabel("Number of orders")
plt.show()
pay_type=payment.groupby('payment_type').aggregate({'order_id':'count'}).rename(columns={'order_id':'count'}).sort_values(by='count',ascending=False).reset_index()
pay_type['perc']=np.round((pay_type['count']/pay_type['count'].sum())*100,2)

plt.figure(figsize=(8,8))
ax=sns.barplot(x='payment_type',y='count',data=pay_type,color='cyan')
plt.title("Mode of Payment")
plt.xlabel('Payment Type')
plt.ylabel('Number of instances')
print("Average value of transaction on credit card : BRL {:,.0f}".format(np.mean(payment[payment.payment_type=='credit_card']['value'])))
print("Average value of transaction on boleto : BRL {:,.0f}".format(np.mean(payment[payment.payment_type=='boleto']['value'])))
print("Average value of transaction on voucher: BRL {:,.0f}".format(np.mean(payment[payment.payment_type=='voucher']['value'])))
print("Average value of transaction on debit card: BRL {:,.0f}".format(np.mean(payment[payment.payment_type=='debit_card']['value'])))
print("Credit Card quantiles")
print(payment[payment.payment_type=='credit_card']['value'].quantile([.01,.25,.5,.75,.99]))
print("")
print("Boleto quantiles")
print(payment[payment.payment_type=='boleto']['value'].quantile([.01,.25,.5,.75,.99]))
print("")
print("Voucher quantiles")
print(payment[payment.payment_type=='voucher']['value'].quantile([.01,.25,.5,.75,.99]))
print("")
print("Debit Card quantiles")
print(payment[payment.payment_type=='debit_card']['value'].quantile([.01,.25,.5,.75,.99]))
plt.figure(figsize=(10,8))
ax=sns.boxplot(x=payment.payment_type,y=payment.value,palette=sns.color_palette(palette="viridis_r"))
ax.set_title("Boxplot for different payment type")
ax.set_xlabel("Transaction type")
ax.set_ylabel("Transaction Value")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)


payment=payment[payment['value']!=0]
plt.figure(figsize=(10,8))
plt.subplot(221)
ax=sns.distplot(np.log(payment[payment.payment_type=='credit_card']['value'])+1,color="red")
ax.set_xlabel("Log Transaction value (BRL)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution plot for credit card transactions")
plt.subplot(222)
ax1=sns.distplot(np.log(payment[payment.payment_type=='boleto']['value'])+1,color="red")
ax1.set_xlabel("Log Transaction value (BRL)")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution plot for boleto transactions")
plt.subplot(223)
ax2=sns.distplot(np.log(payment[payment.payment_type=='debit_card']['value'])+1,color="red")
ax2.set_xlabel("Log Transaction value (BRL)")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution plot for debit card transactions")
plt.subplot(224)
ax3=sns.distplot(np.log(payment[payment.payment_type=='voucher']['value'])+1,color="red")
ax3.set_xlabel("Log Transaction value (BRL)")
ax3.set_ylabel("Frequency")
ax3.set_title("Distribution plot for voucher transactions")


plt.subplots_adjust(wspace = 0.5, hspace = 0.5,
                    top = 1.3)

plt.show()
### Joining with the transaction data:
order_pay=pd.merge(order,sellers,how='left',on=['order_id','product_id'])
order_pay.shape
plt.figure(figsize=(18,6))
ax=sns.barplot(order_pay['seller_id'].value_counts()[:15].index,order_pay['seller_id'].value_counts()[:15].values,palette='Set2')
ax.set_title('Top 15 sellers in Olist')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()
top_15=order_pay.groupby('seller_id').apply(lambda x:x['product_category_name'].unique()).to_frame().reset_index()
top_15.columns=['seller_id','products']
top_15['product_count']=[len(c) for c in top_15['products']]
top_15.sort_values(by='product_count',ascending=False,inplace=True)
top_15.head(15)