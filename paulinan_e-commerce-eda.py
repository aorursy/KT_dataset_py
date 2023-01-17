import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
%matplotlib inline 
import seaborn as sns
import datetime
import scipy.stats as stats
# Reading data:
customers = pd.read_csv("../input/olist_customers_dataset.csv")
geoloc = pd.read_csv("../input/olist_geolocation_dataset.csv")
items = pd.read_csv("../input/olist_order_items_dataset.csv")
payments = pd.read_csv("../input/olist_order_payments_dataset.csv")
reviews = pd.read_csv("../input/olist_order_reviews_dataset.csv")
orders = pd.read_csv("../input/olist_orders_dataset.csv")
print("Customers database contains", customers.shape[0], "rows and", customers.shape[1], "columns.")
customers.head()
cust = customers["customer_unique_id"].nunique()
print("Number of unique customers:",cust)
cities = customers["customer_city"].nunique()
c1 = customers.groupby('customer_city')['customer_id'].nunique().sort_values(ascending=False)
print("There are",cities,"unique cities in the database. The TOP 10 cities are:")
c2 = c1.head(10)
print(c2)
print("\nTOP 10 cities covers", round(c2.sum()/customers.shape[0]*100,1),"percent of all the orders.")
plt.figure(figsize=(16,8))
c2.plot(kind="bar",rot=0)
zips = customers.groupby('customer_zip_code_prefix')['customer_id'].nunique().sort_values(ascending=False)
print("Products were delivered the most frequently,",zips.iloc[0],"times, to the", zips.index[0], "zip code.")
payments.head()
payments["payment_value"].describe()
orders.head()
#orders.dtypes
orders_mod = orders.copy()
orders_mod["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], format='%Y-%m-%d %H:%M:%S')
orders_mod["order_delivered_carrier_date"] = pd.to_datetime(orders["order_delivered_carrier_date"], format='%Y-%m-%d %H:%M:%S')
orders_mod["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"], format='%Y-%m-%d %H:%M:%S')
orders_mod["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"], format='%Y-%m-%d %H:%M:%S')
orders_mod.dtypes
#new.head()
counts = orders_mod.set_index("order_purchase_timestamp").groupby(pd.Grouper(freq='D')).count()
fig = plt.figure(figsize=(16,8))
ax = fig.gca()
counts.plot(y = "order_id", use_index=True, ax=ax)
print(payments.describe())
print(payments["payment_type"].value_counts())
credit = payments[payments["payment_type"]=="credit_card"]
fig, ax = plt.subplots(figsize=(16,8))
credit.hist(column = "payment_installments", bins = credit["payment_installments"].max(), ax=ax)
plt.xlabel("Number of installments")
plt.ylabel("Counts")
plt.title("Histogram of installments count")
mean = payments["payment_value"].mean()
std = payments["payment_value"].std()
skew = payments["payment_value"].skew()
kurt = payments["payment_value"].kurtosis()

text1 = '$\mu=$' + str(round(mean,2))
text2 = '$\sigma=$' +str(round(std,2))
text3 = '$skewness=$' + str(round(skew,2))
text4 = '$kurtosis=$' + str(round(kurt,2))
text = text1 + "\n" + text2 + "\n" + text3 + "\n" + text4

q95 = payments["payment_value"].quantile(.95)
payments_q95 = payments[payments["payment_value"]<q95]

payments_q95.hist(column = "payment_value", bins = 100, figsize=(15,8), rwidth=0.9)
plt.axvline(mean, color='k', linestyle='--')
plt.text(mean+10, 2900, text, fontsize=12)
plt.xlabel("Payment value")
plt.ylabel("Counts")
plt.title("Histogram of payments values (cut at 95th quantile)")
fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(aspect="equal"))
explode = (0.1, 0, 0, 0)
colors = ['#f45a5a', '#449dfc', '#93f96d', '#f9c86d']
legend = ["Credit Card", "Boleto", "Voucher", "Debit Card"]

p = payments["payment_type"][payments["payment_type"] != "not_defined"].value_counts()
p.plot(kind="pie", legend=False, labels=None, startangle=0, explode=explode, autopct='%1.0f%%', pctdistance=0.6, shadow=True, textprops={'weight':'bold', 'fontsize':16}, 
       colors=colors, ax=ax)
ax.legend(legend, loc='best', shadow=True, prop={'weight':'bold', 'size':12}, bbox_to_anchor=(0.6, 0, 0.5,1))
plt.title("Paymement methods", fontweight='bold', size=16)
plt.ylabel("")
print(orders.columns)
# Data preparation
def convert(db,columns_list):
    for column in columns_list:
        db[column] = pd.to_datetime(db[column])

orders_mod = orders.copy()
cols = ['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date']

convert(orders_mod,cols)
orders_mod['delivery_time'] = orders_mod['order_delivered_carrier_date'] - orders_mod['order_purchase_timestamp']
q95 = orders_mod['delivery_time'].quantile(.95)

delivered = orders_mod[orders_mod["order_status"]=="delivered"]
delivered.set_index(delivered['order_purchase_timestamp'], inplace = True)

delivered = delivered.sort_index()
delivered["dts"] = delivered["delivery_time"].dt.total_seconds()
delivered = delivered["dts"].resample("D").mean()
delivered = round(delivered/86400,2)
delivered = delivered[delivered<8]
print(delivered.describe())
m_del = delivered.mean()
ax = delivered.plot(figsize=(16,8))
plt.axhline(m_del,color="k",linestyle='--')
plt.xlabel("Date")
plt.ylabel("Mean delivery time in days")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
deltas = delivered - m_del
ax1 = plt.subplots(figsize=(10, 8))
deltas.hist(bins=40, align="left", rwidth=0.9)
plt.rcParams["figure.figsize"] = (5,4)
plt.xlabel("Delivery time delta to mean [days]")
text1 = '$\mu=$' + str(round(deltas.mean(),2))
text2 = '$\sigma=$' +str(round(deltas.std(),2))
text3 = '$skewness=$' + str(round(deltas.skew(),2))
text4 = '$kurtosis=$' + str(round(deltas.kurt(),2))
text = text1 + "\n" + text2 + "\n" + text3 + "\n" + text4
plt.text(1,30,text, fontsize=12)
#Is this a normal distribution?
ax1 = plt.subplots(figsize=(8, 8))
res = stats.probplot(deltas, dist="norm", plot=plt)
plt.grid(True)