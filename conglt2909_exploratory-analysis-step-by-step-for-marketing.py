import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #collection of command style functions that make matplotlib work
import plotly.express as px #new high-level plotly visualization  library, that exposes a simple syntax for complex chart
import seaborn as sns #statistical data visualization. 
%matplotlib inline  
#Displays output inline

import scipy #to manipulate and visualize data with a wide range of high-level commands.
from scipy.stats.stats import pearsonr
from random import sample #get ramdomly sampling
orders_data=pd.read_csv('../input/brazilian-ecommerce/olist_orders_dataset.csv')
payment_data=pd.read_csv('../input/brazilian-ecommerce/olist_order_payments_dataset.csv')
review_data=pd.read_csv('../input/brazilian-ecommerce/olist_order_reviews_dataset.csv')
items_data=pd.read_csv('../input/brazilian-ecommerce/olist_order_items_dataset.csv')
product_data=pd.read_csv('../input/brazilian-ecommerce/olist_products_dataset.csv')
customers_data=pd.read_csv('../input/brazilian-ecommerce/olist_customers_dataset.csv')
sellers_data=pd.read_csv('../input/brazilian-ecommerce/olist_sellers_dataset.csv')
product_trans_data=pd.read_csv('../input/brazilian-ecommerce/product_category_name_translation.csv')
geoloca_data=pd.read_csv('../input/brazilian-ecommerce/olist_geolocation_dataset.csv')
# Orders_data
round((orders_data.isnull().sum()/len(orders_data)*100),2)
orders_data.order_purchase_timestamp
#Payment_data
round((payment_data.isnull().sum()/len(payment_data)*100),2)
#Review_data
round((review_data.isnull().sum()/len(review_data)*100),2)
#items_data
round((items_data.isnull().sum()/len(items_data)*100),2)
# Product_data
round((product_data.isnull().sum()/len(product_data)*100),2)
# Product_data
round((product_data.isnull().sum()/len(product_data)*100),2)
#Customers_data
round((customers_data.isnull().sum()/len(customers_data)*100),2)
#Sellers_data
round((sellers_data.isnull().sum()/len(sellers_data)*100),2)
#Geolocation_Data
round((geoloca_data.isnull().sum()/len(geoloca_data)*100),2)
#Product_Translation_data
round((product_trans_data.isnull().sum()/len(product_trans_data)*100),2)
#STEP 1:  We use function pd.to_datetime to convert type date columns from object to datetime64
orders_data['order_purchase_timestamp']=pd.to_datetime(orders_data['order_purchase_timestamp'])
orders_data['order_approved_at']=pd.to_datetime(orders_data['order_approved_at'])
orders_data['order_delivered_carrier_date']=pd.to_datetime(orders_data['order_delivered_carrier_date'])
orders_data['order_delivered_customer_date']=pd.to_datetime(orders_data['order_delivered_customer_date'])
orders_data['order_estimated_delivery_date']=pd.to_datetime(orders_data['order_estimated_delivery_date'])
# STEP 2:  We find the average of the time between purchase_timestamp ( not any null) with delivered_carrier.
orders_data_1=orders_data[orders_data['order_delivered_carrier_date'].notnull()]
miss_carrier=(orders_data_1['order_purchase_timestamp']-orders_data_1['order_delivered_carrier_date']).mean()
miss_carrier #we name it: miss_carrier
# After that, we caculate missing time by minus of order_purchase_timestamp and miss_carrier, which we caculated above
added_date=orders_data[orders_data['order_delivered_carrier_date'].isnull()]['order_purchase_timestamp'] - miss_carrier
added_date
#STEP 3: we replace missing values by the new ones.
orders_data['order_delivered_carrier_date']=orders_data['order_delivered_carrier_date'].replace(np.nan,added_date)
orders_data.isnull().sum()
#After treating, there are no-missing values in column:order_delivered_carrier_date
orders_data.isnull().sum()
# STEP 2:  We find the average of the time between purchase_timestamp ( not any null) with delivered_customer_date.
orders_data_2=orders_data[orders_data['order_delivered_carrier_date'].notnull()]
mean_deliver=(orders_data_1['order_purchase_timestamp']-orders_data_1['order_delivered_customer_date']).mean()
mean_deliver #we name it: mean_deliver
# After that, we caculate missing time by minus of order_purchase_timestamp and mean_deliver, which we caculated above
added_date=orders_data[orders_data['order_delivered_customer_date'].isnull()]['order_purchase_timestamp'] - mean_deliver
added_date
#Step 3: we replace missing values by the new ones.
orders_data['order_delivered_customer_date']=orders_data['order_delivered_customer_date'].replace(np.nan,added_date)

#After that, we check missing values in collumn order_delivered_customer_date
orders_data.isnull().sum()
#because the number of missing is too small compared to the amount of rows. So we can drop it
orders_data.dropna(inplace=True)
#There are no-missing values in any column of DataFrame orders_data
orders_data.isnull().sum()
#We notice that missing values in 2 columns review_comment_title and review_comment_message is not worth. Because we have no idea using 2 kind of columns. So we drop it
review_data.isnull().sum()
review_data.drop(columns=['review_comment_title','review_comment_message'],inplace=True)
#There are no missing columns
review_data.isnull().sum()
product_data.dropna(inplace=True)
product_data.isnull().sum()
# Orders_data
orders_data.duplicated(['order_id']).sum()
# Orders_data
items_data.duplicated(['order_id']).sum()
# Orders_data
payment_data.duplicated(['order_id']).sum()
# Orders_data
review_data.duplicated(['order_id']).sum()
# Orders_data
customers_data.duplicated(['customer_id']).sum()
#We merge orders_data with items_data on common column: order_id and inner join 
data_merge=pd.merge(orders_data,items_data,on='order_id',how='inner')

#We merge DataFrame merged with payment_data on common column: order_id and inner join 
data_merge=pd.merge(data_merge,payment_data,on='order_id',how='inner')

#We merge DataFrame merged with review_data on common column: order_id and inner join 
data_merge=pd.merge(data_merge,review_data,on='order_id',how='inner')

#We merge DataFrame merged with product_data on common column: order_id and inner join 
data_merge=pd.merge(data_merge,product_data,on='product_id',how='inner')

#We merge DataFrame merged with customers_data on common column: customer_id and inner join 
data_merge=pd.merge(data_merge,customers_data,on='customer_id',how='inner')

#We merge DataFrame merged with sellers_data on common column: sellers_id and inner join 
data_merge=pd.merge(data_merge,sellers_data,on='seller_id',how='inner')

#We merge DataFrame merged with product_trans_data on common column: product_category_name and inner join 
data_merge=pd.merge(data_merge,product_trans_data,on='product_category_name',how='inner')
#We drop duplicated column in final DataFrame merged and named it Df_ecommerce
Df_ecommerce=data_merge.drop_duplicates(['order_id'])
Df_ecommerce.isnull().sum()
Df_top20prod_rev=Df_ecommerce['price'].groupby(Df_ecommerce['product_category_name_english']).sum().sort_values(ascending=False)[:20]
Df_top20prod_rev
fig=plt.figure(figsize=(16,10))
sns.barplot(y=Df_top20prod_rev.index,x=Df_top20prod_rev.values)
plt.title('Top 20 product category having the largest revenue',fontsize=20)
plt.xlabel('Total revenue',fontsize=17)
plt.ylabel('Product category',fontsize=17)
Df_top20prod_numsell=Df_ecommerce['order_id'].groupby(Df_ecommerce['product_category_name_english']).count().sort_values(ascending=False)[:20]
fig=plt.figure(figsize=(16,10))
sns.barplot(y=Df_top20prod_numsell.index,x=Df_top20prod_numsell.values)
plt.title('Top 20 product category having the largest amount of selling',fontsize=20)
plt.xlabel('Number of selling',fontsize=17)
plt.ylabel('Product category',fontsize=17)
#New column expecting_delivery: time sellers promise delivering for customer.
Df_ecommerce['Expecting_Delivery']=Df_ecommerce['order_estimated_delivery_date'] - Df_ecommerce['order_purchase_timestamp']

#New column Real_delievery: Actual time period customer received after ordering
Df_ecommerce['Real_Delivery']=Df_ecommerce['order_delivered_customer_date'] - Df_ecommerce['order_purchase_timestamp']

#New column Real_Delievery_hour: Convert expecting_delivery to hour, round to 2 decimal
Df_ecommerce['Real_Delivery_hour']=(Df_ecommerce['Real_Delivery']/np.timedelta64(1, 'h')).round(2)

#New column Expecting_delievery_hour: Convert Expecting_delivery to hour, round to 2 decimal
Df_ecommerce['Expecting_delivery_hour']=(Df_ecommerce['Expecting_Delivery']/np.timedelta64(1,'h')).round(2)

#New column Delivery_evaluate: Compare between Expecting and Real Delivery
Df_ecommerce['delivery_evaluate']=round(((2*Df_ecommerce['Expecting_Delivery']-Df_ecommerce['Real_Delivery'])/Df_ecommerce['Expecting_Delivery'])*100,2)
Df_ecommerce
#List of top 20 product categories having the largest revenue
Df_top20prod_rev
# DataFrame of top 20 product categories having the largest revenue
Df_ecommerce_top20rev=Df_ecommerce[Df_ecommerce['product_category_name_english'].isin(Df_top20prod_rev.index)]
#Average delivery of each product category
Df_avg_del_top20=(Df_ecommerce_top20rev['Real_Delivery_hour'].groupby(Df_ecommerce_top20rev['product_category_name_english']).mean()/24).round(2)
fig=plt.figure(figsize=(16,10))
sns.barplot(x=Df_avg_del_top20.values,y=Df_avg_del_top20.index)
plt.title('Average days of  delivery of top 20 product categories',fontsize=20,weight='bold')
plt.tick_params(axis='both',labelsize=13) #stick label 
plt.ylabel('Product category',fontsize=15) #title and fontsize for ylabel
plt.xlabel('Days of delivery',fontsize=15)  #title and fontsize for xlabel
#Get customer_city based on total revenue ( sum of price)
Top5city_rev=Df_ecommerce['price'].groupby(Df_ecommerce['customer_city']).sum().sort_values(ascending=False)[:5]

#Crate new index "other cities" euqally to the rest of other cities
Top5city_rev['other cities']=Df_ecommerce['price'].groupby(Df_ecommerce['customer_city']).sum().sort_values(ascending=False)[5:].sum()

#Create DataFrame and rename 
Top5city_rev=pd.DataFrame(data=Top5city_rev).rename(columns={'price':'Total revenue'})

#Calling new DataFrame
Top5city_rev
Top5city_rev.plot.pie(y='Total revenue',autopct='%1.1f%%',shadow=True,figsize=(10,10),legend=True,textprops={'size': 13},explode=(0.15, 0, 0, 0,0, 0), labeldistance=None,pctdistance=1.1)
plt.legend(loc='lower right',bbox_to_anchor=(1.35,0.5),fontsize=15)
plt.title('Top 5 cities having the largest total revenue',fontsize=20,weight='bold')
plt.show()
#Get customer_city based on amount of selling
Top5city_sellamount=Df_ecommerce['customer_city'].value_counts().sort_values(ascending=False)[:5]

#Create new row: other cities = the rest of other cites
Top5city_sellamount['other cities']=Df_ecommerce['customer_city'].value_counts().sort_values(ascending=False)[5:].sum()

#Create DataFrame and rename column
Top5city_sellamount=pd.DataFrame(data=Top5city_sellamount).rename(columns={'customer_city':'selling amount'})

#Calling new DataFrame
Top5city_sellamount
Top5city_sellamount.plot.pie(y='selling amount',autopct='%1.1f%%',shadow=True,figsize=(10,10),labeldistance=None,textprops={'size':13} ,explode=(0.15, 0, 0, 0, 0, 0),legend=True,pctdistance=1.1)
plt.legend(loc='lower right',bbox_to_anchor=(1.35,0.5),fontsize=15)
plt.title('Top 5 cities having the largest amount of selling',fontsize=20,weight='bold')
plt.show()
#Get new DataFrame of top 20 product categories having the largest revenue
Df_top20prod_review=Df_ecommerce[Df_ecommerce['product_category_name_english'].isin(Df_top20prod_rev.index)]

#Series of average review based on top 20 product categories 
series_top20pro_review=Df_top20prod_review['review_score'].groupby(Df_top20prod_review['product_category_name_english']).mean()

#Calling new series
series_top20pro_review
fig=plt.figure(figsize=(16,10)) #Creating figsize, frame.
sns.barplot(x=series_top20pro_review.values,y=series_top20pro_review.index) 
plt.title('Customer review of top 20 product having the largest revenue',fontsize=20) #title and fontsize of barchart
plt.tick_params(axis='both',labelsize=13) #tick label and font size
plt.xlabel('Review point',fontsize=15)
plt.ylabel('Product categories',fontsize=15)
#Get 20 product categories having the smallest amount of selling
Df_lowest_numsell=Df_ecommerce['order_id'].groupby(Df_ecommerce['product_category_name_english']).count().sort_values(ascending=False)[-20:]
Df_lowest_numsell
#Create new DataFrame just have only 20 product categories having the smallest amount of selling, which we create above
Df_low20prod_review=Df_ecommerce[Df_ecommerce['product_category_name_english'].isin(Df_lowest_numsell.index)]

#Create sereis of average review point of 20 product categories having the lowest amount of selling
series_low20pro_review=Df_low20prod_review['review_score'].groupby(Df_low20prod_review['product_category_name_english']).mean()
series_low20pro_review
# Create barchart 
fig=plt.figure(figsize=(16,10))
sns.barplot(x=series_low20pro_review.values,y=series_low20pro_review.index)
plt.title('Customer review of top 20 product having the lowest amount of selling',fontsize=20)
plt.tick_params(axis='both',labelsize=13)
Df_rev_month=Df_ecommerce[['price']].groupby([Df_ecommerce['product_category_name_english'],Df_ecommerce['order_purchase_timestamp'].map(lambda x: x.strftime('%B'))]).sum().unstack(1).droplevel(axis=1,level=0)
Df_rev_month
#Getting new DataFrames total revenue of top 10 product categories in each month
Df_jan=Df_rev_month['January'].sort_values(ascending=False)[:10]
Df_feb=Df_rev_month['February'].sort_values(ascending=False)[:10]
Df_mar=Df_rev_month['March'].sort_values(ascending=False)[:10]
Df_apr=Df_rev_month['April'].sort_values(ascending=False)[:10]
Df_may=Df_rev_month['May'].sort_values(ascending=False)[:10]
Df_jun=Df_rev_month['June'].sort_values(ascending=False)[:10]
Df_jul=Df_rev_month['July'].sort_values(ascending=False)[:10]
Df_aug=Df_rev_month['August'].sort_values(ascending=False)[:10]
Df_sep=Df_rev_month['September'].sort_values(ascending=False)[:10]
Df_oct=Df_rev_month['October'].sort_values(ascending=False)[:10]
Df_nov=Df_rev_month['November'].sort_values(ascending=False)[:10]
Df_dec=Df_rev_month['December'].sort_values(ascending=False)[:10]

#Create figsize and subplots
f, axes = plt.subplots(6,2, figsize=(20, 100 ))
plt.subplots_adjust(wspace = 0.4 ) #wspace: wide space

# Create individually barplot by Seaborn
sns.barplot(y=Df_jan.index,x=Df_jan.values,ax=axes[0,0])
sns.barplot(y=Df_feb.index,x=Df_feb.values,ax=axes[1,0])
sns.barplot(y=Df_mar.index,x=Df_mar.values,ax=axes[2,0])
sns.barplot(y=Df_apr.index,x=Df_apr.values,ax=axes[3,0])
sns.barplot(y=Df_may.index,x=Df_may.values,ax=axes[4,0])
sns.barplot(y=Df_jun.index,x=Df_jun.values,ax=axes[5,0])
sns.barplot(y=Df_jul.index,x=Df_jul.values,ax=axes[0,1])
sns.barplot(y=Df_aug.index,x=Df_aug.values,ax=axes[1,1])
sns.barplot(y=Df_sep.index,x=Df_sep.values,ax=axes[2,1])
sns.barplot(y=Df_oct.index,x=Df_oct.values,ax=axes[3,1])
sns.barplot(y=Df_nov.index,x=Df_nov.values,ax=axes[4,1])
sns.barplot(y=Df_dec.index,x=Df_dec.values,ax=axes[5,1])

#Set title and fontsize of each 
axes[0,0].set_title('Jan',fontsize=17)
axes[1,0].set_title('Feb',fontsize=17)
axes[2,0].set_title('Mar',fontsize=17)
axes[3,0].set_title('Apr',fontsize=17)
axes[4,0].set_title('May',fontsize=17)
axes[5,0].set_title('Jun',fontsize=17)
axes[0,1].set_title('Jul',fontsize=17)
axes[1,1].set_title('Aug',fontsize=17)
axes[2,1].set_title('Sep',fontsize=17)
axes[3,1].set_title('Oct',fontsize=17)
axes[4,1].set_title('Nov',fontsize=17)
axes[5,1].set_title('Dec',fontsize=17)

#Remove ylabel of each 
axes[0,0].yaxis.label.set_visible(False)
axes[1,0].yaxis.label.set_visible(False)
axes[2,0].yaxis.label.set_visible(False)
axes[3,0].yaxis.label.set_visible(False)
axes[4,0].yaxis.label.set_visible(False)
axes[5,0].yaxis.label.set_visible(False)
axes[0,1].yaxis.label.set_visible(False)
axes[1,1].yaxis.label.set_visible(False)
axes[2,1].yaxis.label.set_visible(False)
axes[3,1].yaxis.label.set_visible(False)
axes[4,1].yaxis.label.set_visible(False)
axes[5,1].yaxis.label.set_visible(False)

#Bold and custom size tick label of each
axes[0,0].tick_params(axis = 'both',  labelsize = 15)
axes[1,0].tick_params(axis = 'both',  labelsize = 15)
axes[2,0].tick_params(axis = 'both',  labelsize = 15)
axes[3,0].tick_params(axis = 'both',  labelsize = 15)
axes[4,0].tick_params(axis = 'both',  labelsize = 15)
axes[5,0].tick_params(axis = 'both',  labelsize = 15)
axes[0,1].tick_params(axis = 'both',  labelsize = 15)
axes[1,1].tick_params(axis = 'both',  labelsize = 15)
axes[2,1].tick_params(axis = 'both',  labelsize = 15)
axes[3,1].tick_params(axis = 'both',  labelsize = 15)
axes[4,1].tick_params(axis = 'both',  labelsize = 15)
axes[5,1].tick_params(axis = 'both',  labelsize = 15)


#Get top 10 product categories having the largest amount of selling 
Df_top10_list=np.array(Df_top20prod_numsell[:10].index)
Df_top10_list
#split number of selling in each product category, month by groupby function. After that, we unstack level 0 of index
Df_numsell_month=Df_ecommerce[['order_id']].groupby([Df_ecommerce['product_category_name_english'],Df_ecommerce['order_purchase_timestamp'].map(lambda x:x.strftime('%m'))]).count().unstack(0).droplevel(axis=1,level=0)

#Get Series of top 10 product categories having the largest number of selling in each month
Df_top10_numsell_month=Df_numsell_month[Df_top10_list].stack(level=0)
Df_top10_numsell_month

indexs=Df_top10_numsell_month.index.get_level_values(0)
colors=np.array(Df_top10_numsell_month.index.get_level_values(1))
plt.figure(figsize=(16,10))

#Using plotly express library to create interactively multiple line charts of top 10 product categories having the largest number of selling in each month
fig=px.line(Df_top10_numsell_month,x=indexs,y=Df_top10_numsell_month.values,color=colors)

#Using fig.update_layout to make visualization more details and clearly
fig.update_layout(xaxis_title='Month',
                  yaxis_title='number of selling',
                  title_text='Top 10 product category having the largest amount of selling in each month',
                  legend_title_text='Product category',
                 )
fig.show()
#Geta list of 20 sellers having the largest revenue by groupby 
Df_20sellers_revenue=Df_ecommerce['price'].groupby(Df_ecommerce['seller_id']).sum().sort_values(ascending=False)[:20]
#Create barchart by Seaborn
fig=plt.figure(figsize=(16,8))
sns.barplot(y=Df_20sellers_revenue.index,x=Df_20sellers_revenue.values)

#Labeling title, x and y axes
plt.title('Top 20 sellers have the largest revenue',fontsize=20)
plt.xlabel('Total revenue',fontsize=17)
plt.ylabel('Seller ID',fontsize=17)
Df_20sellers_numsell=Df_ecommerce['order_id'].groupby(Df_ecommerce['seller_id']).count().sort_values(ascending=False)[:20]
#Create barchart by Seaborn
fig=plt.figure(figsize=(16,8))
sns.barplot(y=Df_20sellers_numsell.index,x=Df_20sellers_numsell.values)

#Labeling title, x and y axes
plt.title('Top 20 sellers have the largest number of selling',fontsize=20)
plt.xlabel('Number of selling',fontsize=17)
plt.ylabel('Seller ID',fontsize=17)
#Create DataFrame of only top 20 sellers
Df_20sellers_rev=Df_ecommerce[Df_ecommerce['seller_id'].isin(Df_20sellers_numsell.index)]
#Get average review_score of top 20 sellers
Df_20sellers_feedback=Df_20sellers_rev['review_score'].groupby(Df_20sellers_rev['seller_id']).mean()
#Create barchart by Seaborn
fig=plt.figure(figsize=(16,8))
sns.barplot(y=Df_20sellers_feedback.index,x=Df_20sellers_feedback.values)

#Labeling title, x and y axes
plt.title('Customer feedback of top 20 sellers have the largest number of selling',fontsize=20)
plt.xlabel('Average review score',fontsize=17)
plt.ylabel('Seller ID',fontsize=17)
Df_ecommerce['customer_id'].value_counts().value_counts()
weight=Df_ecommerce['product_weight_g']
delivery_relhour=Df_ecommerce['Real_Delivery_hour']
delivery_exphour=Df_ecommerce['Expecting_delivery_hour']
satisfy=Df_ecommerce['review_score']
delivery_evaluate=Df_ecommerce['delivery_evaluate']
payment_type=Df_ecommerce['payment_type']
volumetric=Df_ecommerce['product_length_cm']*Df_ecommerce['product_height_cm']*Df_ecommerce['product_width_cm']
pearsonr_coefficient,p_value=pearsonr(delivery_exphour,weight)
print('Correlation Coefficient %0.3f'%(pearsonr_coefficient))
pearsonr_coefficient,p_value=pearsonr(delivery_exphour,volumetric)
print('Correlation Coefficient %0.3f'%(pearsonr_coefficient))
pearsonr_coefficient,p_value=pearsonr(satisfy,delivery_evaluate)
print('Correlation Coefficient %0.3f'%(pearsonr_coefficient))
#We take 200 sample of each payment type
df1=Df_ecommerce[Df_ecommerce['payment_type']=='credit_card'].sample(n=200)
df2=Df_ecommerce[Df_ecommerce['payment_type']=='boleto'].sample(n=200)
df3=Df_ecommerce[Df_ecommerce['payment_type']=='voucher'].sample(n=200)
df4=Df_ecommerce[Df_ecommerce['payment_type']=='debit_card'].sample(n=200)
Df_sample=pd.concat([df1,df2,df3,df4])
def del_evalu_class(x):
    if x>=150: return 'Very good'
    elif 100<=x<150: return 'Good'
    elif 50<=x<100: return 'Normal'
    elif 0<=x<50: return 'Bad'
    else : return 'very bad'
Df_sample['Delivery classification']=Df_sample['delivery_evaluate'].map(del_evalu_class)
Df_sample['Delivery classification'].value_counts()

satisfy=Df_sample['review_score']
payment_type=Df_sample['payment_type']
Delivery_classification=Df_sample['Delivery classification']
table=pd.crosstab(payment_type,Delivery_classification)
table

from scipy.stats import chi2_contingency
chi2,p,dof,expected=chi2_contingency(table.values)
print('chisquare statistic %0.3f p_value %0.3f'%(chi2,p))
table=pd.crosstab(payment_type,satisfy)
table

from scipy.stats import chi2_contingency
chi2,p,dof,expected=chi2_contingency(table.values)
print('chisquare statistic %0.3f p_value %0.3f'%(chi2,p))