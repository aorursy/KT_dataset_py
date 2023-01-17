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
import missingno as msno

import matplotlib.pyplot as plt

import seaborn as sns

# library for seasonal decomposition

from statsmodels.tsa.seasonal import seasonal_decompose

# for map

import plotly.express as px

# set seaborn style

sns.set_style("white")
df = pd.read_excel("../input/dataanz/ANZ synthesised transaction dataset.xlsx")
# print first five rows

df.head()
# count number of missing values

df.isnull().sum().sort_values(ascending = True)*100/df.shape[0]
# visualize missing values

msno.matrix(df);
df.drop(['bpay_biller_code', 'merchant_code'],axis = 1, inplace=True)
# print first five rows of data

df.head()
# check data types

df.dtypes
df.status = df.status.astype('string')

df.account = df.account.astype('string')

df.currency = df.currency.astype('string')
# split long_lat and convert to numeric

df['long'], df['lat'] = df.long_lat.str.split(' ',1).str

df.drop('long_lat', axis = 1, inplace = True)
# convert to numeric

df['long']  = df['long'].astype('float')

df['lat'] = df['lat'].astype('float')

df.txn_description = df.txn_description.astype('string')
df.merchant_id = df.merchant_id.astype("string")
df.merchant_id.isnull().sum()
df.movement.value_counts()
print(df.movement.dtype)

# convert to string

df.movement = df.movement.astype('string')

df.first_name = df.first_name.astype('string')
df.describe()
df.card_present_flag.value_counts(normalize=True).plot(kind='bar')

plt.title("Card Present Status")

plt.xlabel("Flag")

plt.ylabel("Count");
# calculate percentage

df_p = df.status.value_counts()*100/df.status.count()

df_p.plot(kind='bar')

plt.title("Transaction Status")

plt.xlabel("Flag")

plt.ylabel("Percentage");
account_top = df.account.value_counts().sort_values(ascending=False)

account_top[0:10].plot(kind='bar')

plt.title("Top 10 account by Transaction")

plt.xlabel("Account")

plt.ylabel("Tranaction Count");
# Distribution of Transaction Count

account_top.hist()

plt.title("Distribution of Transaction Count by Account")

plt.xlabel("Transaction Count")

plt.ylabel("Frequency");
# calculate percentage

df_p = df.txn_description.value_counts()*100/df.status.count()

df_p.plot(kind='bar')

plt.title("Transaction Description in Percentage")

plt.xlabel("Transaction Description")

plt.ylabel("Percentage");
account_top = df.merchant_id.value_counts().sort_values(ascending=False)

account_top[0:10].plot(kind='bar')

plt.title("Top 10 Merchant Transaction")

plt.xlabel("Merchant Id")

plt.ylabel("Tranaction Count");
# Distribution of Transaction Count for Merchants

account_top.hist()

plt.title("Distribution of Merchant Transaction")

plt.xlabel("Transaction Count")

plt.ylabel("Frequency");
# Distribution of Balance

df.balance.hist()

plt.title("Distribution of Balance")

plt.xlabel("Balance")

plt.ylabel("Frequency");
# Distribution of age

df.age.hist()

plt.title("Distribution of Age")

plt.xlabel("Age")

plt.ylabel("Frequency");
# Distribution of Amount

df.amount.hist()

plt.title("Distribution of Amount")

plt.xlabel("Amount")

plt.ylabel("Frequency");
# calculate percentage

df_p = df.gender.value_counts()/df.gender.count()

df_p.plot(kind='bar')

plt.title("Transaction by Gender")

plt.xlabel("Gender")

plt.ylabel("Percent");
account_top = df.merchant_suburb.value_counts().sort_values(ascending=False)

account_top[0:10].plot(kind='bar')

plt.title("Top 10 Merchant merchant_suburb")

plt.xlabel("merchant_suburb")

plt.ylabel("Count");

account_top = df.merchant_state.value_counts().sort_values(ascending=False)

account_top[0:10].plot(kind='bar')

plt.title("Top 10 Merchant merchant state")

plt.xlabel("merchant_state")

plt.ylabel("Count");
# calculate movement

df_p = df.movement.value_counts()*100/df.movement.count()

df_p.plot(kind='bar')

plt.title("Transaction by movement")

plt.xlabel("movement")

plt.ylabel("Percent");
# balance

sns.boxplot('balance',data = df)

plt.title("Boxplot of Tranaction Balance")

plt.xlabel("Balance");
# age

sns.boxplot('age',data = df)

plt.title("Boxplot of Age")

plt.xlabel("Age");
# amount

sns.boxplot('amount',data = df)

plt.title("Boxplot of Tranaction Amount")

plt.xlabel("Amount");
df_time = df.groupby('date').size()

df_time.plot()

plt.title("Daily Transaction Volume")

plt.xlabel("Date")

plt.ylabel("Count");
df_tran = df.groupby('date')['amount'].mean()

df_tran.plot()

plt.title("Average Daily Transaction Amount")

plt.xlabel("Date")

plt.ylabel("Amount");
result = seasonal_decompose(df_time, model='additive', period=1)

result.plot()

plt.show()
result = seasonal_decompose(df_tran, model='multiplicative', period=1)

result.plot()

plt.show()
# calculate correlaiton

corr = df.drop(['currency','country'],axis = 1).apply(lambda x: pd.factorize(x)[0]).corr(method='pearson')
# set plot size

sns.set(rc={'figure.figsize':(11.7,8.27)})

mask = np.triu(np.ones_like(corr, dtype=np.bool))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr,center=0,mask=mask,cmap = cmap,

            square=True, linewidths=.5, cbar_kws={"shrink": .5});
# calcualte transaction volume by location

df_volume = df.groupby(['lat','long']).size()

df_volume = pd.DataFrame(df_volume.reset_index())

df_volume.columns = ['lat','long','volume']

df_volume['vs'] = np.log(df_volume.volume/10)

df_volume.head()
fig = px.scatter_mapbox(df_volume, lat="lat", lon="long",color="volume", size="vs", hover_name="volume", hover_data=["volume"],

                        color_discrete_sequence=["fuchsia"], zoom=2.5)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
# calcualte average transaction amount by location

df_amount = df.groupby(['lat','long'])['amount'].mean()

df_amount = pd.DataFrame(df_amount.reset_index())

df_amount.columns = ['lat','long','amount']

df_amount['vs'] = np.log(df_amount.amount/10)
fig = px.scatter_mapbox(df_amount, lat="lat", lon="long", color="amount", size="amount", hover_name="amount",

                        hover_data=["amount"],

                        color_discrete_sequence=["fuchsia"], zoom=2.5)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
fig = px.scatter_mapbox(df, lat="lat", lon="long", color="amount", hover_name="amount",

                        hover_data=["amount"],

                        color_discrete_sequence=["fuchsia"], zoom=2.5)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()