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
#pengenalan dataset

import pandas as pd

dataset = pd.read_csv ('../input/retail/retail_raw_reduced.csv')

print('Ukuran dataset: %d baris dan %d kolom\n' % dataset.shape)

print('lima data teratas :')

print(dataset.head())

#Penambahan Kolom Order Month pada Dataset

import datetime

dataset['order_month'] = dataset['order_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").strftime('%Y-%m'))

print(dataset.head())

#Penambahan Kolom GMV pada Dataset

dataset['gmv'] = dataset['item_price']*dataset['quantity']

print('Ukuran dataset: %d baris dan %d kolom\n' % dataset.shape)

print('Lima data teratas:')

print(dataset.head())



#MEMBUAT AGGREGAT

monthly_amount = dataset.groupby('order_month')['gmv'].sum().reset_index()

print(monthly_amount)

#MEMBUAT PLOT LINE

import matplotlib.pyplot as plt

plt.plot(monthly_amount['order_month'], monthly_amount['gmv'])

plt.show()
#CARA ALTERNATIF MEMAKAI FUNGSI PLOT()

import matplotlib.pyplot as plt

dataset.groupby(['order_month'])['gmv'].sum().plot()

plt.show()

#MENGUBAH FIGURE SIZE

import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))

dataset.groupby(['order_month'])['gmv'].sum().plot()

plt.show()

#MENIMBULKAN TITLE, dan AXIS X dan Y

plt.figure(figsize=(15, 5))

dataset.groupby(['order_month'])['gmv'].sum().plot()

plt.title('monthly GMV Year 2019')

plt.xlabel('order Month')

plt.ylabel('Total GMV')

plt.show()

#CUSTOM TITLE DAN AXIS X DAN Y

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

dataset.groupby(['order_month'])['gmv'].sum().plot()

plt.title('Monthly GMV Year 2019', loc='center', pad=40, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize=15)

plt.ylabel('Total Amount', fontsize=15)

plt.show()

#CUSTOM LINE AND POINT

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

dataset.groupby(['order_month'])['gmv'].sum().plot(color='green', marker='o', linestyle='-.', linewidth=2)

plt.title('Monthly GMV Year 2019', loc='center', pad=40, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize=15)

plt.ylabel('Total Amount', fontsize=15)

plt.show()

#CUSTOM GRID

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

dataset.groupby(['order_month'])['gmv'].sum().plot(color='green', marker='o', linestyle='-.', linewidth=2)

plt.title('Monthly GMV Year 2019', loc='center', pad=40, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize=15)

plt.ylabel('Total Amount', fontsize=15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

plt.show()

#CUSTOM AXIS TICKS

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

dataset.groupby(['order_month'])['gmv'].sum().plot(color='green', marker='o', linestyle='-.', linewidth=2)

plt.title('Monthly GMV Year 2019', loc='center', pad=40, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize=15)

plt.ylabel('Total Amount (in Billions)', fontsize=15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.show()

#MENENTUKAN BATAS MINIMUM

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

dataset.groupby(['order_month'])['gmv'].sum().plot(color='green', marker='o', linestyle='-.', linewidth=2)

plt.title('Monthly GMV Year 2019', loc='center', pad=40, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize=15)

plt.ylabel('Total Amount (in Billions)', fontsize=15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.show()

#MENAMBAHKAN INFORMASI PADA PLOT

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 5))

dataset.groupby(['order_month'])['gmv'].sum().plot(color='green', marker='o', linestyle='-.', linewidth=2)

plt.title('Monthly GMV Year 2019', loc='center', pad=40, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize=15)

plt.ylabel('Total Amount (in Billions)', fontsize=15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.text(0.45, 0.72, 'The GMV increased significantly on October 2019', transform=fig.transFigure, color='red')

plt.show()

#MENSAVE KEDALAM CHART KE DALAM GRAFIK

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 5))

dataset.groupby(['order_month'])['gmv'].sum().plot(color='green', marker='o', linestyle='-.', linewidth=2)

plt.title('Monthly GMV Year 2019', loc='center', pad=40, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize=15)

plt.ylabel('Total Amount (in Billions)', fontsize=15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.text(0.45,0.72, 'The GMV increased significantly on October 2019', transform=fig.transFigure, color='red')

plt.savefig('monthly_gmv.png')

plt.show()

#MENGATUR PARAMETER MENYIMPANG GAMBAR

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 5))

dataset.groupby(['order_month'])['gmv'].sum().plot(color='green', marker='o', linestyle='-.', linewidth=2)

plt.title('Monthly GMV Year 2019', loc='center', pad=40, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize=15)

plt.ylabel('Total Amount (in Billions)', fontsize=15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.text(0.45,0.72, 'The GMV increased significantly on October 2019', transform=fig.transFigure, color='red')

plt.savefig('monthly_gmv.png', quality=95)

plt.show()
#MEMBUAT LINE CHART

# Import library

import datetime

import pandas as pd

import matplotlib.pyplot as plt

# Baca dataset

dataset = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/retail_raw_reduced.csv')

# Buat kolom baru yang bertipe datetime dalam format '%Y-%m'

dataset['order_month'] = dataset['order_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").strftime('%Y-%m'))

# Buat Kolom GMV

dataset['gmv'] = dataset['item_price']*dataset['quantity']



# Buat Multi-Line Chart

dataset.groupby(['order_month','brand'])['gmv'].sum().unstack().plot()

plt.title('Monthly GMV Year 2019 - Breakdown by Brand',loc='center',pad=30, fontsize= 20, color='blue')

plt.xlabel('Order Month', fontsize = 15)

plt.ylabel('Total Amount (in Billions)', fontsize = 15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

labels, locations= plt.yticks()

plt.yticks(label, (label/1000000000).astype(int))

plt.gcf().set_size_inches(10, 5)

plt.tight_layout()

plt.show()
# Import library

import datetime

import pandas as pd

import matplotlib.pyplot as plt

# Baca dataset

dataset = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/retail_raw_reduced.csv')

# Buat kolom baru yang bertipe datetime dalam format '%Y-%m'

dataset['order_month'] = dataset['order_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").strftime('%Y-%m'))

# Buat Kolom GMV

dataset['gmv'] = dataset['item_price']*dataset['quantity']



# Buat Multi-Line Chart

dataset.groupby(['order_month','brand'])['gmv'].sum().unstack().plot()

plt.title('Monthly GMV Year 2019 - Breakdown by Brand',loc='center',pad=30, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize = 15)

plt.ylabel('Total Amount (in Billions)', fontsize = 15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

plt.ylim(ymin=0)

labels, locations= plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.gcf().set_size_inches(10, 5)

plt.tight_layout()

plt.show()
import matplotlib.pyplot as plt

dataset.groupby(['order_month','brand'])['gmv'].sum().unstack().plot()

plt.title('Monthly GMV Year 2019 - Breakdown by Brand',loc='center',pad=30, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize = 15)

plt.ylabel('Total Amount (in Billions)',fontsize = 15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.legend(loc='right', bbox_to_anchor=(1.6, 0.5), shadow=True, ncol=2)

plt.gcf().set_size_inches(12, 5)

plt.tight_layout()
#Kustomisasi Colormap

import matplotlib.pyplot as plt

plt.clf()

dataset.groupby(['order_month','province'])['gmv'].sum().unstack().plot(cmap='Set1')

plt.title('Monthly GMV Year 2019 - Breakdown by Province', loc='center',pad=30, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize=15)

plt.ylabel('Total Amount (in Billions)', fontsize = 15)

plt.grid(color='darkgray', linestyle=':', linewidth = 0.5)

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), shadow=True, ncol=3, title='Province', fontsize=9, title_fontsize=11)

plt.gcf().set_size_inches(10, 5)

plt.tight_layout()

plt.show()
# Buat variabel untuk 5 propinsi dengan GMV tertinggi

top_provinces = (dataset.groupby('province')['gmv']

                        .sum()

                        .reset_index()

                        .sort_values(by='gmv', ascending=False)

                        .head(5))

print(top_provinces)



# Buat satu kolom lagi di dataset dengan nama province_top

dataset['province_top'] = dataset['province'].apply(lambda x: x if (x in top_provinces['province'].to_list()) else 'other')



# Plot multi-line chartnya

import matplotlib.pyplot as plt

dataset.groupby(['order_month','province_top'])['gmv'].sum().unstack().plot(marker='.',cmap='plasma')

plt.title('Monthly GMV Year 2019 - Breakdown by Province', loc='center',pad=30, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize=15)

plt.ylabel('Total Amount (in Billions)', fontsize = 15)

plt.grid(color='darkgray', linestyle=':', linewidth = 0.5)

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1), shadow=True, ncol=1)

plt.gcf().set_size_inches(12, 5)

plt.tight_layout()

plt.show()
#MENAMBAHKAN ANOTASI

import matplotlib.pyplot as plt

dataset.groupby(['order_month','province_top'])['gmv'].sum().unstack().plot(marker='.', cmap='plasma')

plt.title('Monthly GMV Year 2019 - Breakdown by Province',loc='center',pad=30, fontsize=20, color='blue')

plt.xlabel('Order Month', fontsize = 15)

plt.ylabel('Total Amount (in Billions)',fontsize = 15)

plt.grid(color='darkgray', linestyle=':', linewidth=0.5)

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1), shadow=True, ncol=1)

# Anotasi pertama

plt.annotate('GMV other meningkat pesat', xy=(5, 900000000),

			 xytext=(4, 1700000000), weight='bold', color='red', 

			 arrowprops=dict(arrowstyle='fancy', 

							 connectionstyle="arc3", 

							 color='red'))









# Anotasi kedua

plt.annotate('DKI Jakarta mendominasi', xy=(3, 3350000000),

			 xytext=(0, 3700000000), weight='bold', color='red', 

			 arrowprops=dict(arrowstyle='->', 

							 connectionstyle="angle", 

							 color='red'))

plt.gcf().set_size_inches(12, 5)

plt.tight_layout()

plt.show()
#MEMBUAT SUBSET   DATA

dataset_dki_q4= dataset[(dataset['province']=='DKI Jakarta') & (dataset['order_month'] >= '2019-10')]

print(dataset_dki_q4.head())

#MEMBUAT PIE CHART

import matplotlib.pyplot as plt

gmv_per_city_dki_q4 = dataset_dki_q4.groupby('city')['gmv'].sum().reset_index()

plt.figure(figsize=(6,6))

plt.pie(gmv_per_city_dki_q4['gmv'], labels = gmv_per_city_dki_q4['city'], autopct='%1.2f%%')

plt.title('GMV Contribution Per City - DKI Jakarta in Q4 2019', loc='center',pad=30, fontsize=15, color='blue')

plt.show()
import matplotlib.pyplot as plt

plt.clf()

dataset_dki_q4.groupby('city')['gmv'].sum().sort_values(ascending=False).plot(kind='bar',color='green')

plt.title('GMV Per City - DKI Jakarta in Q4 2019',loc='center',pad=30, fontsize=15, color='blue')

plt.xlabel('City', fontsize = 15)

plt.ylabel('Total Amount (in Billions)', fontsize = 15)

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.xticks(rotation=0)

plt.show()
#membuat MULTI BAR CHART

import matplotlib.pyplot as plt

dataset_dki_q4.groupby(['city','order_month'])['gmv'].sum().unstack().plot(kind='bar')

plt.title('GMV Per City, Breakdown by Month\nDKI Jakarta in Q4 2019', loc='center', pad=30, fontsize=15, color='blue')

plt.xlabel('Province', fontsize= 12)

plt.ylabel('total Amount (in Billions)', fontsize= 12)

plt.legend(bbox_to_anchor=(1,1), shadow=True, title='Month')

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
#STACKEDBAR

import matplotlib.pyplot as plt

dataset_dki_q4.groupby(['order_month','city'])['gmv'].sum().sort_values(ascending=False).unstack().plot(kind='bar', stacked=True)

plt.title('GMV Per Month. Breakdown by City\nDKI Jakarta in Q4 2019',loc='center',pad=30, fontsize=15, color='blue')

plt.xlabel('Order Month', fontsize= 12)

plt.ylabel('Total Amount (in Billions)', fontsize = 12)

plt.legend(bbox_to_anchor=(1, 1), shadow=True, ncol=1, title='City')

plt.ylim(ymin=0)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000000).astype(int))

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()
#Membuat Agregat Data Customer

data_per_customer = (dataset_dki_q4.groupby('customer_id')

                                   .agg({'order_id':'nunique', 

                                         'quantity': 'sum', 

                                         'gmv':'sum'})

                                   .reset_index()

                                   .rename(columns={'order_id':'orders'}))

print(data_per_customer.sort_values(by='orders',ascending=False))

import matplotlib.pyplot as plt

plt.clf()

# Histogram pertama

plt.figure()

plt.hist(data_per_customer['orders'])

plt.show()

# Histogram kedua

plt.figure()

plt.hist(data_per_customer['orders'], range=(1,5))

plt.title('Distribution of Number of Orders per Customer\nDKI Jakarta in Q4 2019',fontsize=15, color='blue')

plt.xlabel('Number of Orders', fontsize = 12)

plt.ylabel('Number of Customers', fontsize = 12)

plt.show()
#HISTOGRAM PART 2

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.hist(data_per_customer['quantity'], bins=100, range=(1,200), color='brown')

plt.title('Distribution of Total Quantity per Customer\nDKI Jakarta in Q4 2019',fontsize=15, color='blue')

plt.xlabel('Quantity', fontsize= 12)

plt.ylabel('Number of Customers',fontsize= 12)

plt.xlim(xmin=0,xmax=200)

plt.show()

#HISTOGRAM PART 3

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.hist(data_per_customer['gmv'], bins=100, range=(1,200000000), color='green')

plt.title('Distribution of Total GMV per Customer\nDKI Jakarta in Q4 2019', fontsize=15, color='blue')

plt.xlabel('GMV (in Millions)', fontsize= 12)

plt.ylabel('Number of Customers', fontsize =12)

plt.xlim(xmin=0,xmax=200000000)

labels, locations = plt.xticks()

plt.xticks(labels, (labels/1000000).astype(int))

plt.show()
#SCATTER PLOT

import matplotlib.pyplot as plt

plt.clf()

# Scatterplot pertama

plt.figure()

plt.scatter(data_per_customer['quantity'], data_per_customer['gmv'])

plt.show()

# Scatterplot kedua: perbaikan scatterplot pertama

plt.figure(figsize=(10,8))

plt.scatter(data_per_customer['quantity'], data_per_customer['gmv'], marker='+', color='red')

plt.title('Correlation of Quantity and GMV per Customer\nDKI Jakarta in Q4 2019', fontsize=15, color='blue')

plt.xlabel('Quantity', fontsize=12)

plt.ylabel('GMV (in Millions)', fontsize= 12)

plt.xlim(xmin=0,xmax=300)

plt.ylim(ymin=0,ymax=150000000)

labels, locations = plt.yticks()

plt.yticks(labels, (labels/1000000).astype(int))

plt.show()