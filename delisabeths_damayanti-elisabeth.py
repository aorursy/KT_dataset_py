import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

# https://towardsdatascience.com/data-visualization-using-seaborn-fc24db95a850

import seaborn as sns; sns.set()

import os
# List Data

list_dir = os.listdir("/kaggle/input/uisummerschool")

list_dir
data_marketing = pd.read_csv("/kaggle/input/uisummerschool/Marketing.csv", encoding='UTF-8')

data_marketing
data_marketing['Date'] = pd.to_datetime(data_marketing['Date'])
data_marketing.describe()
# KESIMPULAN: Total pengeluaran marketing online dan offline sama.
data_marketing.info() # get DataFrame general info
data_marketing_offline = data_marketing.groupby(data_marketing['Date'].dt.strftime('%B'))['Offline Spend'].sum().sort_values(ascending=False).reset_index()
print("5 bulan pengeluaran marketing offline terbanyak")

data_marketing_offline.head()
data_marketing_online = data_marketing.groupby(data_marketing['Date'].dt.strftime('%B'))['Online Spend'].sum().sort_values(ascending=False).reset_index()
print("5 bulan pengeluaran marketing online terbanyak")

data_marketing_online.head()
data_marketing_offline_2 = data_marketing.groupby(data_marketing['Date'].dt.strftime('%m'))['Offline Spend'].sum().reset_index()

data_marketing_offline_2
data_marketing_online_2 = data_marketing.groupby(data_marketing['Date'].dt.strftime('%m'))['Online Spend'].sum().reset_index()

data_marketing_online_2
mounth_series_marketing = data_marketing_online_2

mounth_series_marketing["Offline Spend"] = data_marketing_offline_2["Offline Spend"]

mounth_series_marketing
ax = sns.lineplot(x="Date", y="Online Spend", data=mounth_series_marketing)

ax = sns.lineplot(x="Date", y="Offline Spend", data=mounth_series_marketing)

ax.set(xlabel='month', ylabel='Spends')
offline_sale = pd.read_csv("/kaggle/input/uisummerschool/Offline_sales.csv", encoding='UTF-8')

offline_sale
offline_sale.head()

offline_sale_2 = offline_sale.groupby(['StockCode']).Quantity.sum().reset_index()

offline_sale_2 = offline_sale_2.sort_values(by= ["Quantity"], ascending=False).reset_index().drop(columns=['index'])

offline_sale_3 = offline_sale_2.head()

offline_sale_3
offline_sale_3.plot(kind='bar', x='StockCode');
online_sale = pd.read_csv("/kaggle/input/uisummerschool/Online_sales.csv", encoding='UTF-8')

online_sale
key_foreign = pd.read_csv("/kaggle/input/uisummerschool/Product.csv", encoding='UTF-8')

key_foreign
# Yang ingin saya analisis adalah comparasi dari marketing dan penjualan.

# Yang saya liat sejauh ini kah marketing terbanyak di online marketing.

# Apakah marketing tersebut meningkatkan jumlah barang yang terjual? Jika dibandingkan per bulan apakah marketing yang genjor meningkatkan penjualan?

# Apakah kegiatan tersebut meningkatkan revenue juga?



# Ataukah marketing itu malah membuat defisit? Jika iya strategi apa yang perlu diambil? Dari printil2an si penjualan.