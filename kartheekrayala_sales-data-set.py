import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import pickle

import datetime as dt

import seaborn as sns

sns.set()

from os import listdir

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



from sklearn.manifold import TSNE

from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn import preprocessing, model_selection, metrics, feature_selection
print(listdir("../input"))
sales = pd.read_excel("../input/sales-data/sales_data.xlsx")
sales.isnull().sum()
sales.rename(columns = {'transaction id': 'Transaction_Id', 'product id':'Product_Id','product description':'Description',

                        'unit price':'Unit_Price','customer id':'Customer_Id','transaction country':'country',

                        'quantity sold':'quantity','transaction timestamp':'transaction_timestamp'},inplace = True)
sales.dropna(axis = 0, subset = ['Product_Id','Unit_Price','country'],inplace = True)
print('Dupplicate entries: {}'.format(sales.duplicated().sum()))

sales.drop_duplicates(inplace = True)
sales.head()
sales.isnull().sum()/sales.shape[0]*100
sales[sales.Description.isnull() == True].head()
sales.dropna(axis = 0,subset = ['Description'],inplace = True)
sales.head()
sales.shape
sales.Transaction_Id.nunique()
sales = sales[sales['Product_Id']!= 'POST']

sales = sales[sales['Product_Id']!= 'D']

sales = sales[sales['Product_Id']!= 'C2']

sales = sales[sales['Product_Id']!= 'M']

sales = sales[sales['Product_Id']!= 'BANK CHARGES']

sales = sales[sales['Product_Id']!= 'PADS']

sales = sales[sales['Product_Id']!= 'DOT']
sales.shape
sales.Transaction_Id = sales.Transaction_Id.astype(str)
sales["IsCancelled"]=np.where(sales.Transaction_Id.apply(lambda l: len(l)== 7), True, False)

sales.IsCancelled.value_counts() /sales.shape[0] * 100
Damaged = sales[(sales.Customer_Id.isnull() == True) & (sales.Unit_Price == 0.0) ].copy()

Damaged.head()
Damaged.shape
Damaged.describe()
sales.Product_Id.nunique()
Clean_sales = sales[~((sales.Customer_Id.isnull() == True) & (sales.Unit_Price == 0))]
Clean_sales.head()
Clean_sales.shape
Promotions = Clean_sales[(Clean_sales.IsCancelled == False) & (Clean_sales.quantity < 0) | (Clean_sales.Unit_Price == 0)].copy()
Promotions.shape
Cancelled_orders = Clean_sales[(Clean_sales['IsCancelled'] == True) & (Clean_sales.Unit_Price != 0)].copy()
Cancelled_orders.describe()
Cancelled_orders.shape
Clean_sales = Clean_sales[Clean_sales['IsCancelled'] == False].copy()
Clean_sales.shape
Clean_sales.Product_Id.nunique()
Clean_sales.Customer_Id.nunique()
Clean_sales = Clean_sales[(Clean_sales['IsCancelled'] == False) & (Clean_sales['quantity'] > 0) & (Clean_sales.Unit_Price >0)].copy()
Clean_sales.describe()
Clean_sales.shape
Guest_sales = Clean_sales[Clean_sales.Customer_Id.isnull() == True].copy()
sales['Product_Id'].nunique()
Clean_sales['Product_Id'].nunique()
Clean_sales = Clean_sales[Clean_sales.Customer_Id.isnull() == False].copy()
Clean_sales['Product_Id'].nunique()
Clean_sales['Revenue'] = Clean_sales['quantity']*Clean_sales['Unit_Price']
Guest_sales['Revenue'] = Guest_sales['quantity']*Guest_sales['Unit_Price']
Clean_sales['Revenue'].sum()/Guest_sales['Revenue'].sum()
Clean_sales.describe()
Guest_sales.describe()
T_Customers = Clean_sales.Customer_Id.nunique()
Clean_sales.groupby('Customer_Id').Product_Id.nunique().value_counts()/T_Customers
description_counts = Clean_sales.Description.value_counts().sort_values(ascending=False).iloc[0:30]

plt.figure(figsize=(20,5))

sns.barplot(description_counts.index, description_counts.values, palette="Purples_r")

plt.ylabel("Counts")

plt.title("Which product descriptions are most common?");

plt.xticks(rotation=90);
Product_Per_person_count = Clean_sales.groupby('Customer_Id').Product_Id.nunique().value_counts()

plt.figure(figsize=(20,5))

sns.barplot(Product_Per_person_count.index, Product_Per_person_count.values)

plt.ylabel("Counts")

plt.title("How many products are brought by a customer?");

plt.xticks(rotation=90);
customer_counts = Clean_sales.Customer_Id.value_counts().sort_values(ascending=False).iloc[0:20] 

plt.figure(figsize=(20,5))

sns.barplot(customer_counts.index, customer_counts.values, order=customer_counts.index)

plt.ylabel("Counts")

plt.xlabel("Customer_Id")

plt.title("Which customers are most common?");
Revenue_Product = Clean_sales.groupby('Product_Id').Revenue.sum()

Revenue_Product = Revenue_Product.sort_values(ascending = False).iloc[0:20]

plt.figure(figsize=(20,5))

sns.barplot(Revenue_Product.index,Revenue_Product.values,order = Revenue_Product.index)

plt.ylabel("Revenue")

plt.title("Top 20 Products");

plt.xticks(rotation=90);
Clean_sales.country.nunique()
country_counts = Guest_sales.country.value_counts().sort_values(ascending=False).iloc[0:20]

plt.figure(figsize=(20,5))

sns.barplot(country_counts.index, country_counts.values)

plt.ylabel("Counts")

plt.title("Which countries made the most transactions?");

plt.xticks(rotation=90);

plt.yscale("log")
country_counts = Clean_sales.country.value_counts().sort_values(ascending=False).iloc[0:20]

plt.figure(figsize=(20,5))

sns.barplot(country_counts.index, country_counts.values)

plt.ylabel("Counts")

plt.title("Which countries made the most transactions?");

plt.xticks(rotation=90);

plt.yscale("log")
Clean_sales.loc[Clean_sales.country=="United Kingdom"].shape[0] / Clean_sales.shape[0] * 100
Clean_sales.Unit_Price.describe()
upper_quantity = np.quantile(Guest_sales.quantity,0.95)
upper_price = np.quantile(Guest_sales.Unit_Price,0.95)
Guest_sales = Guest_sales[(Guest_sales.Unit_Price < upper_price) & (Guest_sales.quantity < upper_quantity)].copy()
Clean_sales.describe()
Guest_sales.describe()
Clean_sales['Revenue'].sum()/Guest_sales['Revenue'].sum()
Guest_sales.Transaction_Id.nunique()
Guest_sales['Revenue'].sum()/(Clean_sales['Revenue'].sum()+Guest_sales['Revenue'].sum())
Clean_sales['Revenue'].sum()/(Clean_sales['Revenue'].sum()+Guest_sales['Revenue'].sum())
16174/(16174+19309)*100
Clean_sales.Transaction_Id.nunique()
Clean_sales.quantity.describe()
Clean_sales["Year"] = Clean_sales.transaction_timestamp.dt.year

Clean_sales["Quarter"] = Clean_sales.transaction_timestamp.dt.quarter

Clean_sales["Month"] = Clean_sales.transaction_timestamp.dt.month

Clean_sales["Week"] = Clean_sales.transaction_timestamp.dt.week

Clean_sales["Weekday"] = Clean_sales.transaction_timestamp.dt.weekday

Clean_sales["Day"] = Clean_sales.transaction_timestamp.dt.day

Clean_sales["Dayofyear"] = Clean_sales.transaction_timestamp.dt.dayofyear

Clean_sales["Date"] = pd.to_datetime(Clean_sales[['Year', 'Month', 'Day']])
grouped_features = ["Date", "Year", "Quarter","Month", "Week", "Weekday", "Dayofyear", "Day",

                    "Product_Id"]
daily_data = pd.DataFrame(Clean_sales.groupby(grouped_features).quantity.sum(),columns=["quantity"])

daily_data["Revenue"] = Clean_sales.groupby(grouped_features).Revenue.sum()

daily_data = daily_data.reset_index()

daily_data.shape
daily_data.loc[:, ["quantity", "Revenue"]].describe()
Cancelled_orders.Transaction_Id.nunique()
low_quantity = daily_data.quantity.quantile(0.01)

high_quantity = daily_data.quantity.quantile(0.99)

print((low_quantity, high_quantity))
low_revenue = daily_data.Revenue.quantile(0.01)

high_revenue = daily_data.Revenue.quantile(0.99)

print((low_revenue, high_revenue))
daily_data = daily_data.loc[

    (daily_data.quantity >= low_quantity) & (daily_data.quantity <= high_quantity)]

daily_data = daily_data.loc[

    (daily_data.Revenue >= low_revenue) & (daily_data.Revenue <= high_revenue)]
fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(daily_data.quantity.values, kde=True, ax=ax[0], color="Orange", bins=30);

sns.distplot(np.log(daily_data.quantity.values), kde=False, ax=ax[1], color="Orange", bins=5);

ax[0].set_xlabel("Number of daily product sales");

ax[0].set_ylabel("Frequency");

ax[0].set_title("How many products are sold per day?");
fig, ax = plt.subplots(1,3,figsize=(20,5))



weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

yearmonth = ["Dec-2010", "Jan-2011", "Feb-2011", "Mar-2011", "Apr-2011", "May-2011",

             "Jun-2011", "Jul-1011", "Aug-2011", "Sep-2011", "Oct-2011", "Nov-2011", 

             "Dec-2011"]

Day_of_Month = [i+1 for i in range(31)]

daily_data.groupby("Weekday").quantity.sum().plot(

    ax=ax[0], marker='o', label="Quantity", c="darkorange");

ax[0].legend();

ax[0].set_xticks(np.arange(0,7))

ax[0].set_xticklabels(weekdays);

ax[0].set_xlabel("")

ax[0].set_title("Total sales per weekday");



ax[1].plot(daily_data.groupby(["Year", "Month"]).quantity.sum().values,

    marker='o', label="Quantities", c="darkorange");

ax[1].set_xticklabels(yearmonth, rotation=90)

ax[1].set_xticks(np.arange(0, len(yearmonth)))

ax[1].legend();

ax[1].set_title("Total sales per month");



ax[2].plot(daily_data.groupby(["Day"]).quantity.sum().values,

    marker='o', label="Quantities", c="darkorange");

ax[2].set_xticklabels(Day_of_Month, rotation=90)

ax[2].set_xticks(np.arange(0, len(Day_of_Month)))

ax[2].legend();

ax[2].set_title("Total sales day wise");
fig, ax = plt.subplots(1,3,figsize=(20,5))



weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

yearmonth = ["Dec-2010", "Jan-2011", "Feb-2011", "Mar-2011", "Apr-2011", "May-2011",

             "Jun-2011", "Jul-1011", "Aug-2011", "Sep-2011", "Oct-2011", "Nov-2011", 

             "Dec-2011"]



daily_data.groupby("Weekday").Revenue.sum().plot(

    ax=ax[0], marker='o', label="Revenue", c="darkorange");

ax[0].legend();

ax[0].set_xticks(np.arange(0,7))

ax[0].set_xticklabels(weekdays);

ax[0].set_xlabel("")

ax[0].set_title("Total Revenue per weekday");



ax[1].plot(daily_data.groupby(["Year", "Month"]).Revenue.sum().values,

    marker='o', label="Revenue", c="darkorange");

ax[1].set_xticklabels(yearmonth, rotation=90)

ax[1].set_xticks(np.arange(0, len(yearmonth)))

ax[1].legend();

ax[1].set_title("Total Revenue per month");



ax[2].plot(daily_data.groupby(["Day"]).Revenue.sum().values,

    marker='o', label="Quantities", c="darkorange");

ax[2].set_xticklabels(Day_of_Month, rotation=90)

ax[2].set_xticks(np.arange(0, len(Day_of_Month)))

ax[2].legend();

ax[2].set_title("Total Revenue day wise");
Clean_sales['Revenue'].sum()/Guest_sales['Revenue'].sum()
Guest_sales["Year"] = Guest_sales.transaction_timestamp.dt.year

Guest_sales["Quarter"] = Guest_sales.transaction_timestamp.dt.quarter

Guest_sales["Month"] = Guest_sales.transaction_timestamp.dt.month

Guest_sales["Week"] = Guest_sales.transaction_timestamp.dt.week

Guest_sales["Weekday"] = Guest_sales.transaction_timestamp.dt.weekday

Guest_sales["Day"] = Guest_sales.transaction_timestamp.dt.day

Guest_sales["Dayofyear"] = Guest_sales.transaction_timestamp.dt.dayofyear

Guest_sales["Date"] = pd.to_datetime(Guest_sales[['Year', 'Month', 'Day']])

Guest_sales.shape
daily_data_guest = pd.DataFrame(Guest_sales.groupby(grouped_features).quantity.sum(),columns=["quantity"])

daily_data_guest["Revenue"] = Guest_sales.groupby(grouped_features).Revenue.sum()

daily_data_guest= daily_data.reset_index()

daily_data_guest.shape
daily_data_guest.loc[:, ["quantity", "Revenue"]].describe()
low_quantity = daily_data_guest.quantity.quantile(0.01)

high_quantity = daily_data_guest.quantity.quantile(0.99)

print((low_quantity, high_quantity))
Clean_sales.Product_Id.nunique()
Guest_sales.Product_Id.nunique()
Cancelled_orders.Product_Id.nunique()
Damaged.Product_Id.nunique()
low_revenue = daily_data_guest.Revenue.quantile(0.01)

high_revenue = daily_data_guest.Revenue.quantile(0.99)

print((low_revenue, high_revenue))
daily_data_guest = daily_data_guest.loc[

    (daily_data_guest.quantity >= low_quantity) & (daily_data_guest.quantity <= high_quantity)]

daily_data_guest = daily_data_guest.loc[

    (daily_data_guest.Revenue >= low_revenue) & (daily_data_guest.Revenue <= high_revenue)]
fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(daily_data_guest.quantity.values, kde=True, ax=ax[0], color="Orange", bins=30);

sns.distplot(np.log(daily_data_guest.quantity.values), kde=False, ax=ax[1], color="Orange", bins=5);

ax[0].set_xlabel("Number of daily product sales");

ax[0].set_ylabel("Frequency");

ax[0].set_title("How many products are sold per day?");


fig, ax = plt.subplots(1,3,figsize=(20,5))



weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

yearmonth = ["Dec-2010", "Jan-2011", "Feb-2011", "Mar-2011", "Apr-2011", "May-2011",

             "Jun-2011", "Jul-1011", "Aug-2011", "Sep-2011", "Oct-2011", "Nov-2011", 

             "Dec-2011"]

Day_of_Month = [i+1 for i in range(31)]

daily_data_guest.groupby("Weekday").quantity.sum().plot(

    ax=ax[0], marker='o', label="Quantity", c="darkorange");

ax[0].legend();

ax[0].set_xticks(np.arange(0,7))

ax[0].set_xticklabels(weekdays);

ax[0].set_xlabel("")

ax[0].set_title("Total sales per weekday");



ax[1].plot(daily_data_guest.groupby(["Year", "Month"]).quantity.sum().values,

    marker='o', label="Quantities", c="darkorange");

ax[1].set_xticklabels(yearmonth, rotation=90)

ax[1].set_xticks(np.arange(0, len(yearmonth)))

ax[1].legend();

ax[1].set_title("Total sales per month");



ax[2].plot(daily_data_guest.groupby(["Day"]).quantity.sum().values,

    marker='o', label="Quantities", c="darkorange");

ax[2].set_xticklabels(Day_of_Month, rotation=90)

ax[2].set_xticks(np.arange(0, len(Day_of_Month)))

ax[2].legend();

ax[2].set_title("Total sales day wise");
fig, ax = plt.subplots(1,3,figsize=(20,5))



weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

yearmonth = ["Dec-2010", "Jan-2011", "Feb-2011", "Mar-2011", "Apr-2011", "May-2011",

             "Jun-2011", "Jul-1011", "Aug-2011", "Sep-2011", "Oct-2011", "Nov-2011", 

             "Dec-2011"]



daily_data_guest.groupby("Weekday").Revenue.sum().plot(

    ax=ax[0], marker='o', label="Revenue", c="darkorange");

ax[0].legend();

ax[0].set_xticks(np.arange(0,7))

ax[0].set_xticklabels(weekdays);

ax[0].set_xlabel("")

ax[0].set_title("Total Revenue per weekday");



ax[1].plot(daily_data_guest.groupby(["Year", "Month"]).Revenue.sum().values,

    marker='o', label="Revenue", c="darkorange");

ax[1].set_xticklabels(yearmonth, rotation=90)

ax[1].set_xticks(np.arange(0, len(yearmonth)))

ax[1].legend();

ax[1].set_title("Total Revenue per month");



ax[2].plot(daily_data_guest.groupby(["Day"]).Revenue.sum().values,

    marker='o', label="Quantities", c="darkorange");

ax[2].set_xticklabels(Day_of_Month, rotation=90)

ax[2].set_xticks(np.arange(0, len(Day_of_Month)))

ax[2].legend();

ax[2].set_title("Total Revenue day wise");
Clean_sales.groupby(["Year","Month",'Customer_Id']).Customer_Id.nunique()
Customer_counts = Clean_sales.groupby(["Year","Month",'Customer_Id']).Customer_Id.nunique().groupby(['Year','Month']).sum()

plt.figure(figsize=(20,5))

sns.barplot(Customer_counts.index, Customer_counts.values, order=Customer_counts.index)

plt.ylabel("Counts")

plt.xlabel("Customer_Id")

plt.title("No.of Registered Customers");
Guest_sales.head()
Customer_Counts = Guest_sales.groupby(["Year","Month",'Transaction_Id']).Transaction_Id.nunique().groupby(['Year','Month']).sum()

plt.figure(figsize=(20,5))

sns.barplot(Customer_counts.index, Customer_counts.values, order=Customer_counts.index)

plt.ylabel("No.of Transactions")

plt.xlabel("Month and Year")

plt.title("No.of Guests");
Cancelled_orders.head()
Cancelled_orders.describe()
Cancelled_orders = Cancelled_orders.loc[Cancelled_orders.quantity < 80].copy()
high_quantity = Cancelled_orders.quantity.quantile(0.99)

print(high_quantity)

high_price = Cancelled_orders.Unit_Price.quantile(0.99)

print(high_price)

Cancelled_orders = Cancelled_orders.loc[Cancelled_orders.quantity <= high_quantity]

Cancelled_orders = Cancelled_orders.loc[Cancelled_orders.Unit_Price <= high_price]
Cancelled_orders.describe()
Cancelled_Ids = Cancelled_orders.Customer_Id.value_counts().sort_values(ascending=False).iloc[0:20]

plt.figure(figsize=(20,5))

sns.barplot(Cancelled_Ids.index, Cancelled_Ids.values,order = Cancelled_Ids.index)

plt.ylabel("Counts")

plt.xlabel('Cutomer_Id')

plt.title("Which Customers cancelled the most?");

plt.xticks(rotation=90);
Cancelled_Ids = Cancelled_orders.Product_Id.value_counts().sort_values(ascending=False).iloc[0:20]

plt.figure(figsize=(20,5))

sns.barplot(Cancelled_Ids.index, Cancelled_Ids.values,order = Cancelled_Ids.index)

plt.ylabel("Counts")

plt.xlabel('Product_Id')

plt.title("Which Product are cancelled most?");

plt.xticks(rotation=90);
Cancelled_Ids = Cancelled_orders.Product_Id.value_counts().sort_values(ascending=False)
Damaged_Id = Damaged.Product_Id.value_counts().sort_values(ascending=False).iloc[0:20]

plt.figure(figsize=(20,5))

sns.barplot(Damaged_Id.index,Damaged_Id.values,order = Damaged_Id.index)

plt.ylabel("Counts")

plt.xlabel('Product_Id')

plt.title("Which Product are damaged?");

plt.xticks(rotation=90);
def unique_counts(data):

   for i in data.columns:

       count = data[i].nunique()

       print(i, ": ", count)

unique_counts(Clean_sales)
Clean_sales.drop(["IsCancelled"],axis = 1,inplace = True)
Guest_sales.describe()
unique_counts(Guest_sales)
unique_counts(Cancelled_orders)
sales.Transaction_Id.nunique()
3694/24306
Guest_sales.shape
Clean_sales.shape
Guest_sales.drop(["Customer_Id","IsCancelled"],axis = 1,inplace = True)
Promotions.shape
Promotions.describe()
Promotions.quantity = Promotions.quantity.apply(lambda x: abs(x))
Promotions.describe()
Promotions['Revenue'] = Promotions['Unit_Price']*Promotions['quantity']
Promotions['Revenue'].sum()/Clean_sales['Revenue'].sum()
Clean_sales['Revenue'].sum()+Guest_sales['Revenue'].sum()
Promotions['Revenue'].sum()/(Clean_sales['Revenue'].sum()+Guest_sales['Revenue'].sum())
le = LabelEncoder()

le.fit(Clean_sales['country'])
l = [i for i in range(38)]

dict(zip(list(le.classes_), l))
Clean_sales['country'] = le.transform(Clean_sales['country'])
with open('labelencoder.pickle', 'wb') as g:

    pickle.dump(le, g)
Clean_sales.describe()
Clean_sales['transaction_timestamp'].min()
Clean_sales['transaction_timestamp'].max()
NOW = dt.datetime(2011,12,10)

Clean_sales['transaction_timestamp'] = pd.to_datetime(Clean_sales['transaction_timestamp'])
Clean_sales.head(1)
product_id = Clean_sales[['Product_Id','quantity','Unit_Price']]

product_id.head()
custom_aggregation = {}

custom_aggregation["Product_Id"] = lambda x:len(x)

custom_aggregation["quantity"] = "sum"

custom_aggregation["Unit_Price"] = "median"
product_id = product_id.groupby("Product_Id").agg(custom_aggregation)
product_id.columns = ["orders", "total_Quantity", "Median_UnitPrice"]

product_id.head()
def RScore(x,p,d):

    if x <= d[p][0.20]:

        return 1

    elif x <= d[p][0.40]:

        return 2

    elif x <= d[p][0.60]: 

        return 3

    elif x <= d[p][0.80]:

        return 4

    else:

        return 5
quantiles = product_id.quantile(q=[0.20,0.40,0.60,0.80])

quantiles = quantiles.to_dict()

quantiles
product_id['o_quartile'] = product_id['orders'].apply(RScore, args=('orders',quantiles,))

product_id['tq_quartile'] = product_id['total_Quantity'].apply(RScore, args=('total_Quantity',quantiles,))

product_id['mup_quartile'] = product_id['Median_UnitPrice'].apply(RScore, args=('Median_UnitPrice',quantiles,))

product_id.head()
product_id['cluster'] = product_id['o_quartile']*product_id['tq_quartile']*product_id['mup_quartile']

product_id.head()
product_id.cluster.nunique()
plt.figure(figsize=(20,5))

Product_Id = product_id.cluster.value_counts()

sns.barplot(Product_Id.index,Product_Id.values,order = Product_Id.index)

plt.ylabel("Number of Products")

plt.xlabel('Cluster')

plt.title("Product Segementation");

plt.xticks(rotation=90);
NOW = dt.datetime(2011,12,10)

Clean_sales['transaction_timestamp'] = pd.to_datetime(Clean_sales['transaction_timestamp'])
custom_aggregation = {}

custom_aggregation["transaction_timestamp"] = ["min","max",lambda x: len(x)]

custom_aggregation["Revenue"] = "sum"



custom_aggregation
Clean_sales.shape
Clean_sales.Customer_Id.nunique()
rfmTable = Clean_sales.groupby("Customer_Id").agg(custom_aggregation)

rfmTable.head()
rfmTable.shape
rfmTable.columns = ["min_time", "max_time", "frequency", "monetary_value"]
rfmTable["max_Recency"] = (NOW - rfmTable["min_time"]).dt.days

rfmTable["min_Recency"] = (NOW - rfmTable["max_time"]).dt.days
rfmTable.head(5)
custom_aggregation = {}



custom_aggregation["Recency"] = ["min", "max"]

custom_aggregation["transaction_timestamp"] = lambda x: len(x)

custom_aggregation["Revenue"] = "sum"
rfmTable_final = rfmTable.drop(['min_time','max_time'],axis = 1)
rfmTable_final.head(5)
rfmTable_final.shape
quantiles = rfmTable_final.quantile(q=[0.25,0.5,0.75])

quantiles = quantiles.to_dict()

quantiles
segmented_rfm = rfmTable_final
segmented_rfm.head()
def RScore(x,p,d):

    if x <= d[p][0.25]:

        return 1

    elif x <= d[p][0.50]:

        return 2

    elif x <= d[p][0.75]: 

        return 3

    else:

        return 4

    

def FMScore(x,p,d):

    if x <= d[p][0.25]:

        return 4

    elif x <= d[p][0.50]:

        return 3

    elif x <= d[p][0.75]: 

        return 2

    else:

        return 1
segmented_rfm['r_quartile'] = segmented_rfm['min_Recency'].apply(RScore, args=('min_Recency',quantiles,))

segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))

segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))

segmented_rfm.head()
segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)

segmented_rfm.head()
segmented_rfm[segmented_rfm['RFMScore']=='111'].sort_values('monetary_value', ascending=False)
segmented_rfm.head(5)
segmented_rfm = segmented_rfm.reset_index()
segmented_rfm.head(5)
segmented_rfm['RFMScore'].value_counts()
RFM_score = segmented_rfm['RFMScore'].value_counts().iloc[:30]

plt.figure(figsize=(20,5))

sns.barplot(RFM_score.index, RFM_score.values,order = RFM_score.index)

plt.ylabel("Counts")

plt.xlabel('RFM Score')

plt.title("Type of customers");

plt.xticks(rotation=90);
Clean_sales = pd.merge(Clean_sales,segmented_rfm, on='Customer_Id')
Clean_sales.columns
Clean_sales = Clean_sales.drop(columns=['r_quartile', 'f_quartile', 'm_quartile'])
Clean_sales.head()
product_id.head()
product_id.shape
product_cluster = product_id.drop(['orders','total_Quantity','Median_UnitPrice','o_quartile','tq_quartile','mup_quartile'],axis = 1)

product_cluster = product_cluster.to_dict()
product_id.cluster.nunique()
cluster = Clean_sales['Product_Id'].apply(lambda x : product_cluster['cluster'][x])

cluster
df2 = pd.get_dummies(cluster, prefix="Cluster").mul(Clean_sales["Revenue"], 0)

df2 = pd.concat([Clean_sales['Customer_Id'], df2], axis=1)

df2_grouped = df2.groupby('Customer_Id').sum()
df2_grouped
df2_grouped.shape
df2_grouped_final = df2_grouped.div(df2_grouped.sum(axis=1), axis=0)

df2_grouped_final = df2_grouped_final.fillna(0)
df2_grouped_final.head()
custom_aggregation = {}

custom_aggregation["transaction_timestamp"] = ["min","max",lambda x: len(x)]

custom_aggregation["Revenue"] = ["min","max","mean","sum"]

custom_aggregation["country"] = "median"

custom_aggregation["quantity"] = "sum"

custom_aggregation
df_grouped_final = Clean_sales.groupby("Customer_Id").agg(custom_aggregation)
df_grouped_final.head()
df_grouped_final.columns = ["min_time", "max_time","frequency","min","max","mean","monetary_value","country","quantity"]

df_grouped_final["max_Recency"] = (NOW - rfmTable["min_time"]).dt.days

df_grouped_final["min_Recency"] = (NOW - rfmTable["max_time"]).dt.days
df_grouped_final.head(5)
df_grouped_final.drop(["min_time","max_time"],axis = 1,inplace = True)
df2_grouped_final.head(5)
df_grouped_final.head()
df_grouped_final.shape
df2_grouped_final.shape
X1 = df_grouped_final.to_numpy()

X2 = df2_grouped_final.to_numpy()



scaler = StandardScaler()

X1 = scaler.fit_transform(X1)

X_final_std_scale = np.concatenate((X1, X2), axis=1)
x = list(range(2, 12))

y_std = []

for n_clusters in x:

    print("n_clusters =", n_clusters)

    

    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=10)

    kmeans.fit(X_final_std_scale)

    clusters = kmeans.predict(X_final_std_scale)

    silhouette_avg = silhouette_score(X_final_std_scale, clusters)

    y_std.append(silhouette_avg)

    print("The average silhouette_score is :", silhouette_avg, "with Std Scaling")
kmeans = KMeans(init='k-means++', n_clusters = 8, n_init=30, random_state=0)  # random state just to be able to provide cluster number durint analysis

kmeans.fit(X_final_std_scale)

clusters = kmeans.predict(X_final_std_scale)
plt.figure(figsize = (20,8))

n, bins, patches = plt.hist(clusters, bins=8)

plt.xlabel("Cluster")

plt.title("Number of customers per cluster")

plt.xticks([rect.get_x()+ rect.get_width() / 2 for rect in patches], ["Cluster {}".format(x) for x in range(8)])



for rect in patches:

    y_value = rect.get_height()

    x_value = rect.get_x() + rect.get_width() / 2



    space = 5

    va = 'bottom'

    label = str(int(y_value))

    

    plt.annotate(

        label,                      

        (x_value, y_value),         

        xytext=(0, space),          

        textcoords="offset points", 

        ha='center',                

        va=va)
df_grouped_final["cluster"] = clusters
final_dataset = pd.concat([df_grouped_final, df2_grouped_final], axis = 1)

final_dataset.head()
final_dataset.shape
final_dataset_V2 = final_dataset.reset_index()
final_dataset_V2.to_csv("final_dataset_V2.csv",index=False)
with open('Clean_sales.pickle', 'wb') as f:

    pickle.dump(Clean_sales, f)
tsne = TSNE(n_components=2)

proj = tsne.fit_transform(X_final_std_scale)



plt.figure(figsize=(5,5))

plt.scatter(proj[:,0], proj[:,1], c=clusters)

plt.title("Visualization of the clustering with TSNE", fontsize="25")
final_dataset[final_dataset['cluster']==0]
final_dataset[final_dataset['cluster']==0].mean()
temp_final_df = final_dataset.reset_index()
cust0 = list(temp_final_df[temp_final_df['cluster']==0]['Customer_Id'])
cluster0 = Clean_sales[Clean_sales['Customer_Id'].isin(cust0)]

pd.DataFrame(cluster0[['quantity', 'Unit_Price', 'Revenue', 'frequency', 'min_Recency'

         , 'monetary_value']].mean())
cluster0['Description'].value_counts()[:20]

RFM_score = cluster0['Description'].value_counts()[:10]

plt.figure(figsize=(5,5))

sns.barplot(RFM_score.index, RFM_score.values,order = RFM_score.index)

plt.ylabel("Count")

plt.xlabel('Products')

plt.title("Top 10 products of Cluster0");

plt.xticks(rotation=90);
custom_aggregation = {}

custom_aggregation["country"] = lambda x:x.iloc[0]

custom_aggregation["RFMScore"] = lambda x:x.iloc[0]



cluster0_grouped = cluster0.groupby("Customer_Id").agg(custom_aggregation)
cluster0_grouped['RFMScore'].value_counts().iloc[:20]

RFM_score = cluster0_grouped['RFMScore'].value_counts().iloc[:20]

plt.figure(figsize=(5,5))

sns.barplot(RFM_score.index, RFM_score.values,order = RFM_score.index)

plt.ylabel("Count")

plt.xlabel('RFm_Score')

plt.title("Top 10 RFMScore of Cluster0");

plt.xticks(rotation=90);
cluster0_grouped['country'].value_counts()
final_dataset[final_dataset['cluster']==6].mean()
cust6 = list(temp_final_df[temp_final_df['cluster']==6]['Customer_Id'])
cluster6 = Clean_sales[Clean_sales['Customer_Id'].isin(cust6)]

cluster6[['quantity', 'Unit_Price', 'Revenue', 'frequency', 'min_Recency'

         , 'monetary_value']].mean()
cluster6['Description'].value_counts()[:10]



RFM_score = cluster6['Description'].value_counts()[:10]

plt.figure(figsize=(5,5))

sns.barplot(RFM_score.index, RFM_score.values,order = RFM_score.index)

plt.ylabel("Count")

plt.xlabel('Products')

plt.title("Top 10 products of Cluster6");

plt.xticks(rotation=90);
custom_aggregation = {}

custom_aggregation["country"] = lambda x:x.iloc[0]

custom_aggregation["RFMScore"] = lambda x:x.iloc[0]



cluster6_grouped = cluster6.groupby("Customer_Id").agg(custom_aggregation)
cluster6_grouped['RFMScore'].value_counts()

RFM_score = cluster6_grouped['RFMScore'].value_counts().iloc[:20]

plt.figure(figsize=(5,5))

sns.barplot(RFM_score.index, RFM_score.values,order = RFM_score.index)

plt.ylabel("Count")

plt.xlabel('RFm_Score')

plt.title("Top 10 RFMScore of Cluster6");

plt.xticks(rotation=90);
cluster6_grouped['country'].value_counts()
final_dataset[final_dataset['cluster']==1].mean()
cust1 = list(temp_final_df[temp_final_df['cluster']==1]['Customer_Id'])
cluster1 = Clean_sales[Clean_sales['Customer_Id'].isin(cust1)]

cluster1[['quantity', 'Unit_Price', 'Revenue', 'frequency', 'min_Recency'

         , 'monetary_value']].mean()
cluster1['Description'].value_counts()[:10]
cluster1['Description'].value_counts()[:20]

RFM_score = cluster1['Description'].value_counts()[:10]

plt.figure(figsize=(5,5))

sns.barplot(RFM_score.index, RFM_score.values,order = RFM_score.index)

plt.ylabel("Count")

plt.xlabel('Products')

plt.title("Top 10 products of Cluster1");

plt.xticks(rotation=90);
custom_aggregation = {}

custom_aggregation["country"] = lambda x:x.iloc[0]

custom_aggregation["RFMScore"] = lambda x:x.iloc[0]



cluster1_grouped = cluster1.groupby("Customer_Id").agg(custom_aggregation)
cluster1_grouped['RFMScore'].value_counts()

RFM_score = cluster1_grouped['RFMScore'].value_counts().iloc[:20]

plt.figure(figsize=(5,5))

sns.barplot(RFM_score.index, RFM_score.values,order = RFM_score.index)

plt.ylabel("Count")

plt.xlabel('RFm_Score')

plt.title("Top 10 RFMScore of Cluster1");

plt.xticks(rotation=90);
cluster1_grouped['country'].value_counts()