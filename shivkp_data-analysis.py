import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
desc = pd.read_csv("../input/DescriptionDataCoSupplyChain.csv")
token = pd.read_csv("../input/tokenized_access_logs.csv")
#Data Description
desc.info()
desc
data = pd.read_csv('../input/DataCoSupplyChainDataset.csv',encoding = "ISO-8859-1")
data.info()
data.head()
df = data.copy(deep=False)

percent_Miss = (df.isnull().sum() / len(df)) * 100
values_missing = df.isnull().sum()
missingValuesDf = pd.DataFrame({'missing values': values_missing,'percentMissing': percent_Miss})
missingValuesDf
data = data.drop(['Days for shipping (real)', 'Days for shipment (scheduled)', 'Customer Email', 'Customer Fname',
                  'Customer Segment', 'Product Image', 'Customer Street', 'Customer Zipcode', 'Order Id', 
                  'Order Zipcode','Order Status','Product Description','Customer Lname','Customer Password',
                  'Product Status','Late_delivery_risk','Order State','Customer State','Product Card Id',
                  'Department Id','shipping date (DateOrders)','Delivery Status','Latitude','Longitude',
                  'Order Item Cardprod Id','Order Item Id','Order Customer Id'], axis=1)
data.head()
ln = data.copy(deep=False)
set_len = []
for column in ln.columns:
    set_len.append(len(set(ln[column])))
    
length_of_set = pd.DataFrame({"Column":ln.columns, "Length of set": set_len})
length_of_set
data.shape
data.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
data.info()
data.describe()
# change object data into int or float

hist_data = data.copy(deep=False)
# drop unused data
hist_data = hist_data.drop(['Category Id','Customer Id','order date (DateOrders)','Product Category Id',
                            'Order Country','Order City',], axis=1)
from sklearn.preprocessing import LabelEncoder
def Change_obj_type(data):
    for column in data.columns:
        if data[column].dtype == type(object):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
    return data
customer_data = Change_obj_type(hist_data)
customer_data
#overview of data Distribution
customer_data.hist(alpha=0.8, figsize=(12, 10))
plt.tight_layout()
plt.show()
# Customer Country
sns.countplot(x='Customer Country', data=data);
plt.title('Distribution of Customer Country');
# Top ten Customer City
data['Customer City'].value_counts()[:10].plot(kind='bar')
plt.title("Top 10 Customer city")
# Payment type
plot = sns.countplot(x='Type', data=data);
plot.set_xticklabels(plot.get_xticklabels(), rotation=60)
#data['Type'].value_counts().plot(kind='bar')
plt.title("Distribution of Payment types")
# Top 15 Category of goods
data['Category Name'].value_counts()[:15].plot(kind='bar')
plt.title("Top 15 Category of goods")
# Department Name
plt.figure(figsize=(6,4))
plot = sns.countplot(x='Department Name', data=data);
plot.set_xticklabels(plot.get_xticklabels(), rotation=60)
plt.title("Departments")

#data['Department Name'].value_counts().plot(kind='bar')
# Market
sns.countplot(x='Market', data=data);

#data['Market'].value_counts().plot(kind='bar')
plt.title("Market list")
# Order Region
plt.figure(figsize=(7,4))
plot = sns.countplot(x='Order Region', data=data);
plot.set_xticklabels(plot.get_xticklabels(), rotation=60)
#data['Order Region'].value_counts().plot(kind='bar')
plt.title("Order Regions")
# Top 20 Product Name
data['Product Name'].value_counts()[:20].plot(kind='bar')
plt.title('Top 20 Product Name')
# Shipping Mode
sns.countplot(x='Shipping Mode', data=data)
plt.title('Shipping Mode')
plt.xticks(rotation = 90)
plt.show()
corr_data = data.copy(deep=False)
# drop unused data
corr_data = corr_data.drop(['Category Id','Customer Id','order date (DateOrders)','Product Category Id'], axis=1)

# change object type data into int
from sklearn.preprocessing import LabelEncoder
def Change_obj_type(data):
    for column in data.columns:
        if data[column].dtype == type(object):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
    return data
cor_data = Change_obj_type(corr_data)
from scipy.stats import norm
corr_m = cor_data.corr()
f, ax = plt.subplots(figsize=(25,18))
plot = sns.heatmap(corr_m, ax = ax,annot = True, cmap ="YlGnBu", linewidths = 0.1)
plt.xlabel('xlabel', fontsize=18)
#plot.figure.savefig("output.png")
#

plt.figure(figsize=(12, 12))
plt.subplot(3,3,1)
sns.boxplot(x = 'Order Item Discount Rate', data=cor_data)
plt.subplot(3,3,2)
sns.boxplot(x = 'Benefit per order', data=cor_data)
plt.subplot(3,3,3)
sns.boxplot(x = 'Sales per customer', data=cor_data)
plt.subplot(3,3,4)
sns.boxplot(x = 'Category Name', data=cor_data)
plt.subplot(3,3,5)
sns.boxplot(x = 'Customer Country', data=cor_data)
plt.subplot(3,3,6)
sns.boxplot(x = 'Department Name', data=cor_data)
plt.subplot(3,3,7)
sns.boxplot(x = 'Order Item Discount', data=cor_data)
plt.subplot(3,3,8)
sns.boxplot(x = 'Product Price', data=cor_data)
plt.subplot(3,3,9)
sns.boxplot(x = 'Order Item Quantity', data=cor_data)
plt.show()
new_data = data.drop(['Order Profit Per Order','Order Item Product Price','Order Item Total'], axis=1)
new_data.info()
new_data['order date (DateOrders)'].min(), new_data['order date (DateOrders)'].max()
import datetime as dt
PRASENT_DATE = dt.datetime(2017, 9, 10)
new_data['order date (DateOrders)'] = pd.to_datetime(new_data['order date (DateOrders)'])
new_data.head()
new_data.info()
new_data['TotalPrice'] = new_data['Order Item Quantity'] * new_data['Product Price']
rfm = new_data.groupby('Customer Id').agg({'order date (DateOrders)':lambda date:(PRASENT_DATE - date.max()).days,
                                           'Order Item Quantity':lambda num:len(num),
                                           'TotalPrice':lambda price:price.sum()})
rfm.columns
# Change the name of columns
rfm.columns=['monetary','frequency','recency']
rfm['recency'] = rfm['recency'].astype(int)
rfm[rfm['monetary']<0]=0
rfm = rfm.drop_duplicates()
rfm
rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, ['1','2','3','4'])
rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4','3','2','1'])
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1'])
rfm.head()
rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
rfm.head()
# Filter out Top/Best cusotmers
rfm.sort_values(['RFM_Score'],axis=0, ascending=True).head()