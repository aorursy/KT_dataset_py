import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns
dat = pd.read_csv("../input/ecommerce-data/data.csv", encoding= 'unicode_escape')

pd.set_option('display.max_columns', 500)

dat.info()
dat.info()
dat.head(15)
dat['InvoiceDate'] = pd.to_datetime(dat['InvoiceDate'])
dat.CustomerID = dat.CustomerID.astype(object)
dat.info() 
dat.isnull().sum()
dat.isnull().sum() / dat.shape[0] * 100
pd.set_option('display.max_rows', 1000)
null_data = dat[dat.isnull().any(axis=1)]
null_data.info()
null_data = dat[dat['Description'].isnull()]
null_data.sample(15)
dat = dat.dropna()
dat.isnull().sum()
dat["IsCancelled"]=np.where(dat.InvoiceNo.apply(lambda l: l[0]=="C"), True, False)
dat.IsCancelled.value_counts() / dat.shape[0] * 100
dat.loc[dat.IsCancelled==True].describe()
dat = dat.loc[dat.IsCancelled==False].copy()
dat = dat.drop("IsCancelled", axis=1)
dat.StockCode.nunique(), dat.Description.nunique()
stockcode_frequency = dat.StockCode.value_counts().sort_values(ascending=False)
description_frequency = dat.Description.value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(2,1,figsize=(20,15))
sns.barplot(stockcode_frequency.iloc[0:19].index,
            stockcode_frequency.iloc[0:19].values,
            ax = ax[0], palette="Blues_r")
ax[0].set_ylabel("Frequency")
ax[0].set_xlabel("Stockcode")
ax[0].set_title("Which stockcodes are most common?");
sns.barplot(description_frequency.iloc[0:19].index,
            description_frequency.iloc[0:19].values,
            ax = ax[1], palette="Purples_r")
ax[1].set_ylabel("Frequency")
ax[1].set_xlabel("Description")
ax[1].tick_params(labelrotation=90)

ax[1].set_title("Which Descriptions are most common?");
customer_frequency = dat.CustomerID.value_counts().sort_values(ascending=False).iloc[0:19] 
plt.figure(figsize=(19,10))
customer_frequency.index = customer_frequency.index.astype('Int64') 
sns.barplot(customer_frequency.index, customer_frequency.values, order=customer_frequency.index, palette="Spectral_r")
plt.ylabel("Frequency")
plt.xlabel("CustomerID")
plt.title("Which customers are most common?");
country_frequency = dat.Country.value_counts().sort_values(ascending=False).iloc[0:20]
plt.figure(figsize=(20,5))
sns.barplot(country_frequency.index, country_frequency.values, palette="plasma_r")
plt.ylabel("Frequency")
plt.title("Which countries made the most transactions?");
plt.xticks(rotation=90);
x = dat.groupby(['CustomerID','Country']).size().sort_values(ascending=False).iloc[0:19]
pd.DataFrame(x)
customer_frequency = dat.CustomerID.value_counts().sort_values(ascending=False).iloc[0:19] 
uk_customers = dat.groupby(dat['CustomerID']).size().where(dat['Country'] == 'United Kingdom').sort_values(ascending=False).iloc[0:19]
fig, ax = plt.subplots(2,1,figsize=(20,20))
sns.barplot(customer_frequency.index,
            customer_frequency.values,
            ax = ax[0], palette="Blues_r", order=customer_frequency.index)
ax[0].set_ylabel("Frequency")
ax[0].set_xlabel("CustomerID")
ax[0].set_title("Which Customers are most common?");
sns.barplot(uk_customers.index,
            uk_customers.values,
            ax = ax[1], palette="cool", order=uk_customers.index)
ax[1].set_ylabel("Frequency")
ax[1].set_xlabel("CustomerID")
ax[1].set_title("Which Customers are most common in the United Kingdom?");
dat.UnitPrice.describe()
dat = dat.loc[dat.UnitPrice > 0].copy()
fig, ax = plt.subplots(2,1,figsize=(15,15))
sns.distplot(dat.UnitPrice, ax=ax[0])
ax[0].set_ylabel('Frequency')
sns.distplot(np.log(dat.UnitPrice), ax=ax[1], bins=20)
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel("Log-Unit-Price");
np.exp(-2),np.exp(3)
dat = dat.loc[(dat.UnitPrice > 0.1) & (dat.UnitPrice < 20)].copy()
dat.UnitPrice.describe()
fig, ax = plt.subplots(2,1,figsize=(15,15))
sns.distplot(dat.UnitPrice, ax=ax[0])
ax[0].set_ylabel('Frequency')
sns.distplot(np.log(dat.UnitPrice), ax=ax[1], bins=20)
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel("Log-Unit-Price");
dat.Quantity.describe()
fig, ax = plt.subplots(2,1,figsize=(15,15))
sns.distplot(dat.Quantity, ax=ax[0], kde=False)
ax[0].set_title("Quantity distribution")
ax[0].set_ylabel('Frequency')
ax[0].set_yscale("log")
sns.distplot(np.log(dat.Quantity), ax=ax[1], bins=20, kde=False)
ax[0].set_title("Log-Quantity distribution")
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel("Log-Quantity");
np.exp(4),np.quantile(dat.Quantity, 0.95)
dat = dat.loc[dat.Quantity < 55].copy()
fig, ax = plt.subplots(2,1,figsize=(15,15))
sns.distplot(dat.Quantity, ax=ax[0], kde=False)
ax[0].set_title("Quantity distribution")
ax[0].set_ylabel('Frequency')
ax[0].set_yscale("log")
sns.distplot(np.log(dat.Quantity), ax=ax[1], bins=20, kde=False)
ax[0].set_title("Log-Quantity distribution")
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel("Log-Quantity");
dat.Quantity.describe()
dat["Revenue"] = dat.Quantity * dat.UnitPrice

dat["Month"] = dat.InvoiceDate.dt.month

dat.groupby('Month').sum().sort_values(by='Revenue', ascending=False)
plt.rcParams.update({'font.size': 12})
z = dat.groupby('Month').sum().sort_values(by='Revenue',ascending=False)
x = z.index
y = z['Revenue'].sort_values(ascending=False)
plt.figure(figsize=(10,10))
sns.barplot(x, y, order=x)
plt.ylabel("Revenue", Size=14)
plt.xlabel("Months", Size=14)
plt.title("Which Month had the highest Revenue?", Size=14);

df = dat[['StockCode','Revenue']].groupby('StockCode').sum().sort_values(by='Revenue', ascending=False).iloc[0:9]
df
dat[dat['StockCode'] == '22423'].sample(10)