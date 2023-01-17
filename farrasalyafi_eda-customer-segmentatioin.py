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
#Importing Libraries
#Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Fetaure Selection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#Modelling Algoritm
from sklearn.cluster import KMeans

#Model Evaluation
from yellowbrick.cluster import SilhouetteVisualizer
#Load All The Data
olist_orders = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv')
olist_products = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_products_dataset.csv')
olist_items = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv')
olist_customers = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_customers_dataset.csv')
olist_payments = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv')
olist_sellers = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_sellers_dataset.csv')
olist_geolocation = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_geolocation_dataset.csv')
olist_reviews = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv')
olist_product_category_name = pd.read_csv('/kaggle/input/brazilian-ecommerce/product_category_name_translation.csv') #Untuk menerjemahkan dari bahasa Brazil ke Bahasa Inggris
#Menggabungkan semua data-data 
all_data = olist_orders.merge(olist_items, on='order_id', how='left')
all_data = all_data.merge(olist_payments, on='order_id', how='inner')
all_data = all_data.merge(olist_reviews, on='order_id', how='inner')
all_data = all_data.merge(olist_products, on='product_id', how='inner')
all_data = all_data.merge(olist_customers, on='customer_id', how='inner')
all_data = all_data.merge(olist_sellers, on='seller_id', how='inner')
all_data = all_data.merge(olist_product_category_name,on='product_category_name',how='inner')
#all_data = all_data.merge(olist_geolocation, on='seller_zip_code_prefix', how='inner')
#Melihat berapa persen data yang kosong pad setiap kolomnya
round((all_data.isnull().sum()/ len(all_data)*100),2)
#Melihat info yang ada pada data baik jumlah kolom, input sampai memori
all_data.info()
#Meruba tipe data pada kolom tanggal agar seusai tipe datanya 
date_columns = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
             'order_estimated_delivery_date', 'shipping_limit_date', 'review_creation_date', 'review_answer_timestamp'] 
for col in date_columns:
    all_data[col] = pd.to_datetime(all_data[col], format='%Y-%m-%d %H:%M:%S')
#Melihat apakah ada data yang duplikat
print('Data yang duplikat: ',all_data.duplicated().sum())
#Membuat kolom Month_order untuk data exploration
all_data['Month_order'] = all_data['order_purchase_timestamp'].dt.to_period('M').astype('str')
#Memilih input mulai dari 01-2017 sampai 08-2018
#Karena terdapat data yang kurang seimbang dengan rata-rata setiap bulan pada data sebelum 01-2017 dan setelah 08-2018
#berdasarkan data pembelian / order_purchase_timestamp
start_date = "2017-01-01"
end_date = "2018-08-31"

after_start_date = all_data['order_purchase_timestamp'] >= start_date
before_end_date = all_data['order_purchase_timestamp'] <= end_date
between_two_dates = after_start_date & before_end_date
all_data = all_data.loc[between_two_dates]
#Membagi data berdasarkan tipe datanya
only_numeric = all_data.select_dtypes(include=['int', 'float'])
only_object = all_data.select_dtypes(include=['object'])
only_time = all_data.select_dtypes(include=['datetime', 'timedelta'])
#Melihat berapa persen data yang kosong pad setiap kolomnya
round((all_data.isnull().sum()/ len(all_data)*100),2)
#Menangani input yang kosong pada kolom order_approved_at 
missing_1 = all_data['order_approved_at'] - all_data['order_purchase_timestamp']
print(missing_1.describe())
print('='*50)
print('Median dari waktu order sampai approved: ',missing_1.median())

#kita ambil median karena ada yang langsung di approve dari waktu dia order,ada juga yang sampai 60 hari
add_1 = all_data[all_data['order_approved_at'].isnull()]['order_purchase_timestamp'] + missing_1.median()
all_data['order_approved_at']= all_data['order_approved_at'].replace(np.nan, add_1)
#Menangani input yang kosong pada kolom order_approved_at 
all_data[['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date']].head()
#Menangani input yang kosong pada kolom order_delivered_carrier_date
missing_2 = all_data['order_delivered_carrier_date'] - all_data['order_approved_at']
print(missing_2.describe())
print('='*50)
print('Median dari waktu apporved sampai dikirim: ',missing_2.median())

#kita ambil median karena ada yang di dikirim dalam 21 jam dari waktu dia approved,ada juga yang sampai 107 hari
add_2 = all_data[all_data['order_delivered_carrier_date'].isnull()]['order_approved_at'] + missing_2.median()
all_data['order_delivered_carrier_date']= all_data['order_delivered_carrier_date'].replace(np.nan, add_2)
#Menangani input yang kosong pada kolom order_delivered_customer_date
missing_3 = all_data['order_delivered_customer_date'] - all_data['order_delivered_carrier_date']
print(missing_3.describe())
print('='*50)
print('Median dari waktu dikirim sampai diterima customer: ',missing_3.median())

#kita ambil median karena ada yang waktu pengiriman dalam -17 Hari berarti dia outliers,ada juga yang waktu pengiriman sampai 205 hari
add_3 = all_data[all_data['order_delivered_customer_date'].isnull()]['order_delivered_carrier_date'] + missing_3.median()
all_data['order_delivered_customer_date']= all_data['order_delivered_customer_date'].replace(np.nan, add_3)
#Menangani kolom review_comment_title dan review_comment_message
#Karena jumlah input yang kosong sangat banyak, dan tidak mungkin untuk diisi dikarenakan tidak ada variabel yang dapat
#digunakan untuk menghitungnya. Karena ini adalah komentar danjudul komentartnya
#Maka kita akan hilangkan kolom tersebut

all_data = all_data.drop(['review_comment_title', 'review_comment_message'], axis=1)
#Menangani input kosong pada kolom product_weight_g, product_length_cm, product_height_cm, product_width_cm
#Karena jumlahnya hanya 1, maka kita drop saja
all_data = all_data.dropna()
#Cek kembali apakah masih ada input yang kosong
round((all_data.isnull().sum()/len(all_data)*100),2)
#Menyesuaikan tipe data dengan input datanya
all_data = all_data.astype({'order_item_id': 'int64', 
                            'product_name_lenght': 'int64',
                            'product_description_lenght':'int64', 
                            'product_photos_qty':'int64'})
#Membuat kolom order_process_time untuk melihat berapa lama waktu yang dibutuhkan dari mulai order sampai
#barang diterima oleh customer
all_data['order_process_time'] = all_data['order_delivered_customer_date'] - all_data['order_purchase_timestamp']
#Membuat kolom order_delivery_time untuk melihat berapa lama waktu pengiriman yang dibutuhkan tiap order
all_data['order_delivery_time'] = all_data['order_delivered_customer_date'] - all_data['order_delivered_carrier_date']
#Membuat kolom order_time_accuracy untuk melihat apakah dari estimasi waktu sampai ada yang sesuai atau terlambat
#Jika nilainya + positive, maka dia lebih cepat sampai, jika 0 maka dia tepat waktu, namun jika - negatif maka dia terlambat
all_data['order_accuracy_time'] = all_data['order_estimated_delivery_date'] - all_data['order_delivered_customer_date'] 
#Membuat kolom order_approved_time untuk melihat berapa lama waktu yang dibutuhkan mulai dari order sampai approved
all_data['order_approved_time'] = all_data['order_approved_at'] - all_data['order_purchase_timestamp'] 
#Membuat kolom review_send_timeuntuk mengetahui berapa lama waktu dikirimnya survey kepuasan setelah barang diterima
all_data['review_send_time'] = all_data['review_creation_date'] - all_data['order_delivered_customer_date']
#Membuat kolom review_answer_time untuk mengetahui berapa lama waktu yang dibutuhkan untuk mengisi review setelah
#dikirim survey kepuasan pelanggan.
all_data['review_answer_time'] = all_data['review_answer_timestamp'] - all_data['review_creation_date']
#Menggabungkan kolom product_length_cm, product_height_cm, dan product_width_cm untuk membuatnya menjadi volume
#dengan kolom baru yaitu product_volume
all_data['product_volume'] = all_data['product_length_cm'] * all_data['product_height_cm'] * all_data['product_width_cm']
#Produk apa yang paling laris?
top_20_product_best_seller = all_data['order_item_id'].groupby(all_data['product_category_name_english']).sum().sort_values(ascending=False)[:20]
#print(top_20_product_best_seller)

#Kita plot untuk visualisasinya
fig=plt.figure(figsize=(16,9))
sns.barplot(y=top_20_product_best_seller.index,x=top_20_product_best_seller.values)
plt.title('Top 20 Most Selling Product',fontsize=20)
plt.xlabel('Total Product Sold',fontsize=17)
plt.ylabel('Product category',fontsize=17)
#Kota mana yang paling banyak belanja?
top_20_city_shopping = all_data['order_item_id'].groupby(all_data['customer_city']).sum().sort_values(ascending=False)[:20]
#print(top_20_city_shopping)

#Kita plot untuk visualisasinya
fig=plt.figure(figsize=(16,9))
sns.barplot(y=top_20_city_shopping.index,x=top_20_city_shopping.values)
plt.title('Top 20 Most City Shopping',fontsize=20)
plt.xlabel('Total Product',fontsize=17)
plt.ylabel('City',fontsize=17)
#Siapa customer paling banyak belanja berdasarkan jumlah order?
top_10_customer_shopping = all_data['order_item_id'].groupby(all_data['customer_id']).count().sort_values(ascending=False)[:10]
#print(top_10_customer_shopping)

#Kita plot untuk visualisasinya
fig=plt.figure(figsize=(16,9))
sns.barplot(y=top_10_customer_shopping.index,x=top_10_customer_shopping.values)
plt.title('Top 10 Customer Based on Order Amount',fontsize=20)
plt.xlabel('Amount of Product',fontsize=17)
plt.ylabel('Customer ID',fontsize=17)
#Siapa customer yang paling banyak pengeluaranya dalam belanja berdasarkan harga?
top_10_customer_shopping = all_data['payment_value'].groupby(all_data['customer_id']).sum().sort_values(ascending=False)[:10]
#print(top_10_customer_shopping)

#Kita plot untuk visualisasinya
fig=plt.figure(figsize=(16,9))
sns.barplot(y=top_10_customer_shopping.index,x=top_10_customer_shopping.values)
plt.title('Top 10 Customer Based on Spending',fontsize=20)
plt.xlabel('Spending Amount',fontsize=17)
plt.ylabel('Customer ID',fontsize=17)
#Seller mana yang paling banyak jual?
top_10_seller_order = all_data['order_item_id'].groupby(all_data['seller_id']).sum().sort_values(ascending=False)[:10]
#print(top_10_seller_order)

#Kita plot untuk visualisasinya
fig=plt.figure(figsize=(16,9))
sns.barplot(y=top_10_seller_order.index,x=top_10_seller_order.values)
plt.title('Top 10 Seller Base on Sold Product',fontsize=20)
plt.xlabel('Total Product',fontsize=17)
plt.ylabel('Seller ID',fontsize=17)
#Seller mana yang paling banyak penghasilan berdasarkan revenue?
top_10_seller_order = all_data['price'].groupby(all_data['seller_id']).sum().sort_values(ascending=False)[:10]
#print(top_10_seller_order)

#Kita plot untuk visualisasinya
fig=plt.figure(figsize=(16,9))
sns.barplot(y=top_10_seller_order.index,x=top_10_seller_order.values)
plt.title('Top 10 Seller Based on Revenue',fontsize=20)
plt.xlabel('Amount of Revenue',fontsize=17)
plt.ylabel('Seller ID',fontsize=17)
#Seller mana yang paling banyak penghasilan berdasarkan revenue?
top_10_seller_order = all_data[all_data['review_score'] == 5].groupby(all_data['seller_id']).sum().sort_values(by=['review_score'],ascending=False)[:10]
#print(top_10_seller_order)

#Kita plot untuk visualisasinya
fig=plt.figure(figsize=(16,9))
sns.barplot(y=top_10_seller_order.index,x=top_10_seller_order.review_score)
plt.title('Top 10 Seller Based on Review Score',fontsize=20)
plt.xlabel('Amount of Revenue',fontsize=17)
plt.ylabel('Seller ID',fontsize=17)
#Sebaran status order customer
round(all_data.order_status.value_counts() / len(all_data),2)
#Berapa median waktu dari order sampai diterima yang dibutuhkan dalam setiap order perbulanya?
order_time_by_month = all_data['order_process_time'].groupby(all_data['Month_order']).median(numeric_only=False) #masukan argumen numeric_only untuk menghitung timedelta

#Membuat visualisasinya
fig=plt.figure(figsize=(16,9))
plt.plot(order_time_by_month.index, order_time_by_month.values, marker='o')
plt.title('Median Order Time By Month',fontsize=20)
plt.xlabel('Month',fontsize=17)
plt.xticks(#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
#         ['January', 'February', 'March','April', 'Mei', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
          rotation=90)
plt.ylabel('Time (Day)',fontsize=17)
#Berapa median waktu pengiriman yang dibutuhkan dalam setiap order perbulanya?
delivery_time_by_month = all_data['order_delivery_time'].groupby(all_data['Month_order']).median(numeric_only=False) #masukan argumen numeric_only untuk menghitung timedelta

#Membuat visualisasinya
fig=plt.figure(figsize=(16,9))
plt.plot(delivery_time_by_month.index, delivery_time_by_month.values / 86400, marker='o')
plt.title('Median Delivery Time By Month',fontsize=20)
plt.xlabel('Month',fontsize=17)
plt.xticks(#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
         # ['January', 'February', 'March','April', 'Mei', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
          rotation=90)
plt.ylabel('Time (Day)',fontsize=17)
#Berapa median akurasi waktu dari estimasi pengiriman dan samapi customer dalam setiap order perbulanya?
accuracy_time_by_month = all_data['order_accuracy_time'].groupby(all_data['Month_order']).median(numeric_only=False) #masukan argumen numeric_only untuk menghitung timedelta

#Membuat visualisasinya
fig=plt.figure(figsize=(16,9))
plt.plot(accuracy_time_by_month.index, accuracy_time_by_month.values / 86400, marker='o')
plt.title('Median Accuracy Time By Month',fontsize=20)
plt.xlabel('Month',fontsize=17)
plt.xticks(#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          #['January', 'February', 'March','April', 'Mei', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
          rotation=90)
plt.ylabel('Time (Day)',fontsize=17)
#Berapa median lama waktu sampai diapproved dari waktu order dalam setiap order perbulanya?
approved_time_by_month = all_data['order_approved_time'].groupby(all_data['Month_order']).median(numeric_only=False) #masukan argumen numeric_only untuk menghitung timedelta

#Membuat visualisasinya
fig=plt.figure(figsize=(16,9))
plt.plot(approved_time_by_month.index, approved_time_by_month.values / 60, marker='o')
plt.title('Median Approved Time By Month',fontsize=20)
plt.xlabel('Month',fontsize=17)
plt.xticks(#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
         # ['January', 'February', 'March','April', 'Mei', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
          rotation=90)
plt.ylabel('Time (Minutes)',fontsize=17)
#10 kategori produk dengan waktu tercepat mulai dari order sampai diterima customer
order_time_by_category = pd.DataFrame(all_data['order_process_time'].groupby(all_data['product_category_name_english']).median(numeric_only=False).sort_values(ascending=True)[:10])

#Visualiasi
fig=plt.figure(figsize=(16,9))
sns.barplot(y=order_time_by_category.index, x=order_time_by_category['order_process_time'].dt.days)
plt.title('Top 10 Fastest Product Category Order Time',fontsize=20)
plt.xlabel('Order Time (Day)',fontsize=17)
plt.ylabel('Product Category',fontsize=17)
#10 kategori produk dengan waktu paling lama mulai dari order sampai diterima customer
order_time_by_category = pd.DataFrame(all_data['order_process_time'].groupby(all_data['product_category_name_english']).median(numeric_only=False).sort_values(ascending=False)[:10])

#Visualiasi
fig=plt.figure(figsize=(16,9))
sns.barplot(y=order_time_by_category.index, x=order_time_by_category['order_process_time'].dt.days)
plt.title('Top 10 Slowest Product Category Order Time',fontsize=20)
plt.xlabel('Order Time (Day)',fontsize=17)
plt.ylabel('Product Category',fontsize=17)
#Berapakah Order setiap bulanya?
order_count_by_month = all_data['order_item_id'].groupby(all_data['Month_order']).sum()

#Visualisasi
fig=plt.figure(figsize=(16,9))
sns.barplot(y=order_count_by_month.values, x=order_count_by_month.index, color="Salmon")
plt.title('Monthly Order',fontsize=20)
plt.xlabel('Month',fontsize=17)
plt.xticks(rotation=90)
plt.ylabel('Amount Order',fontsize=17)
#Berapakah Revenue setiap bulanya?
revenue_count_by_month = all_data['payment_value'].groupby(all_data['Month_order']).sum()

#Visualisasi
fig=plt.figure(figsize=(16,9))
sns.barplot(y=revenue_count_by_month.values, x=revenue_count_by_month.index, color="Salmon")
plt.title('Monthly Revenue',fontsize=20)
plt.xlabel('Month',fontsize=17)
plt.xticks(rotation=90)
plt.ylabel('Amount Revenue',fontsize=17)
#Berapaka customer aktif setiap bulanya?
customer_active_by_month = all_data.groupby('Month_order')['customer_unique_id'].nunique().reset_index()

#Visualisasi
fig=plt.figure(figsize=(16,9))
sns.barplot(y=customer_active_by_month['customer_unique_id'], x=customer_active_by_month['Month_order'], color="Salmon")
plt.title('Monthly Active User',fontsize=20)
plt.xlabel('Month',fontsize=17)
plt.xticks(rotation=90)
plt.ylabel('Amount of User',fontsize=17)
#melihat tanggal awal dan terakhir pembelian 
print('Min : {}, Max : {}'.format(min(all_data.order_purchase_timestamp), max(all_data.order_purchase_timestamp)))
#Menghitung RFM
import datetime as dt
pin_date = max(all_data.order_purchase_timestamp) + dt.timedelta(1)

#Membuat dataframe RFM
rfm = all_data.groupby('customer_unique_id').agg({
    'order_purchase_timestamp' : lambda x: (pin_date - x.max()).days,
    'order_item_id' : 'count', 
    'payment_value' : 'sum'})

#Merubah nama kolom
rfm.rename(columns = {'order_purchase_timestamp' : 'Recency', 
                      'order_item_id' : 'Frequency', 
                      'payment_value' : 'Monetary'}, inplace = True)

rfm.head()
#Kita akan menggunakan Inter Quartile Range untuk menangani ouliers
#Menentukan Limit
def limit(i):
    Q1 = rfm[i].quantile(0.5)
    Q3 = rfm[i].quantile(0.95)
    IQR = Q3 - Q1
    
    #menentukan upper limit biasa dan upper limit ekstim
    lower_limit = rfm[i].quantile(0.5) - (IQR * 1.5)
    lower_limit_extreme = rfm[i].quantile(0.5) - (IQR * 3)
    upper_limit = rfm[i].quantile(0.95) + (IQR * 1.5)
    upper_limit_extreme = rfm[i].quantile(0.5) + (IQR * 3)
    print('Lower Limit:', lower_limit)
    print('Lower Limit Extreme:', lower_limit_extreme)
    print('Upper Limit:', upper_limit)
    print('Upper Limit Extreme:', upper_limit_extreme)

#Mengitung persen outliers dari data    
def percent_outliers(i):
    Q1 = rfm[i].quantile(0.5)
    Q3 = rfm[i].quantile(0.95)
    IQR = Q3 - Q1
    
    #menentukan upper limit biasa dan upper limit ekstim
    lower_limit = rfm[i].quantile(0.5) - (IQR * 1.5)
    lower_limit_extreme = rfm[i].quantile(0.5) - (IQR * 3)
    upper_limit = rfm[i].quantile(0.95) + (IQR * 1.5)
    upper_limit_extreme = rfm[i].quantile(0.95) + (IQR * 3)
    #melihat persenan outliers terhadap total data
    print('Lower Limit: {} %'.format(rfm[(rfm[i] >= lower_limit)].shape[0]/ rfm.shape[0]*100))
    print('Lower Limit Extereme: {} %'.format(rfm[(rfm[i] >= lower_limit_extreme)].shape[0]/rfm.shape[0]*100))
    print('Upper Limit: {} %'.format(rfm[(rfm[i] >= upper_limit)].shape[0]/ rfm.shape[0]*100))
    print('Upper Limit Extereme: {} %'.format(rfm[(rfm[i] >= upper_limit_extreme)].shape[0]/rfm.shape[0]*100))
#Melihat outliers pada kolom Recency
sns.boxplot(x=rfm["Recency"])
#Melihat ouliers pada kolom Frequency
sns.boxplot(x=rfm["Frequency"])
#Melihat ouliers pada kolom Monetary
sns.boxplot(x=rfm["Monetary"])
print(limit('Monetary'))
print('-'*50)
print(percent_outliers('Monetary'))
#Menghilangkan outliers pada kolom Monetary yang lebih dari 1500 karena diluar dari 95% batas maksimal atas persebaran data
outliers1_drop = rfm[(rfm['Monetary'] > 1500)].index
rfm.drop(outliers1_drop, inplace=True)
#Membuat group customer berdasarkan Recency, Frequency, dan Monetary
#Karena Recency jika semakin sedikit harinya semakin bagus, maka akan membuat urutanya secara terbalik
r_labels = range(3, 0, -1)
r_groups = pd.qcut(rfm.Recency, q = 3, labels = r_labels).astype('int')

#Karena Frequency sangat banyak pada nilai 1, maka tidak bisa menggunakan qcut, 
#karena nilainya akan condong ke yang paling banyak
f_groups = pd.qcut(rfm.Frequency.rank(method='first'), 3).astype('str')
#rfm['F'] = np.where((rfm['Frequency'] != 1) & (rfm['Frequency'] != 2), 3, rfm.Frequency)

m_labels = range(1, 4)
m_groups = pd.qcut(rfm.Monetary, q = 3, labels = m_labels).astype('int')
#Membuat kolom berdasarkan group yang telah dibuat
rfm['R'] = r_groups.values
rfm['F'] = f_groups.values
rfm['M'] = m_groups.values
rfm['F'].value_counts()
#Merubah input kolom F menjadi categorical
rfm['F'] = rfm['F'].replace({'(0.999, 30871.333]' : 1,
                             '(30871.333, 61741.667]' : 2,
                             '(61741.667, 92612.0]' : 3}).astype('int')
#Menggabungkan ketiga kolom tersebut
rfm['RFM_Segment'] = rfm.apply(lambda x: str(x['R']) + str(x['F']) + str(x['M']), axis = 1)
rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis = 1)
rfm.head()
#Membuat label berdasarkan RFM_Score
score_labels = ['Bronze', 'Silver', 'Gold']
score_groups = pd.qcut(rfm.RFM_Score, q=3, labels = score_labels)
rfm['RFM_Level'] = score_groups.values
rfm.head()
#Visualisasi nilai RFM
fig, ax = plt.subplots(figsize=(16, 9))
plt.subplot(3, 1, 1); sns.distplot(rfm.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm['Frequency'], kde_kws={'bw': 0.1}, label='Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm.Monetary, label = 'Monetary')

plt.tight_layout()
plt.show()
#Membuat distribusi data menjadi normal
from scipy import stats

rfm_log = rfm[['Recency', 'Monetary']].apply(np.log, axis = 1).round(3)
rfm_log['Frequency'] = stats.boxcox(rfm['Frequency'])[0]
rfm_log.head()
#Membuat semua data dala ukuran yang sama dengan cara scaling
scaler = StandardScaler()
minmax = MinMaxScaler()
rfm_scaled = scaler.fit_transform(rfm_log)
#Membuat dataframe baru setelah di-scaling
rfm_scaled = pd.DataFrame(rfm_scaled, index = rfm.index, columns = rfm_log.columns)
rfm_scaled.head()
#Visualisasi kembali RFM setalah log transformasi dan scaling
fig, ax = plt.subplots(figsize=(16, 9))
plt.subplot(3, 1, 1); sns.distplot(rfm_scaled.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm_scaled.Frequency, kde_kws={'bw': 0.1}, label='Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm_scaled.Monetary, label = 'Monetary')

plt.tight_layout()
plt.show()
#Mencari titik optimal cluster dengan ELbow Method
wcss = {}

for i in range(1, 11):
    kmeans = KMeans(n_clusters= i, init= 'k-means++', max_iter= 300)
    kmeans.fit(rfm_scaled)
    wcss[i] = kmeans.inertia_
    
#Visualisasi Elbow Method
fig, ax = plt.subplots(figsize=(16, 9))
sns.pointplot(x = list(wcss.keys()), y = list(wcss.values()))
plt.title('Elbow Method')
plt.xlabel('K Numbers')
plt.ylabel('WCSS')
plt.show()
#Memilih n_clusters = 4 sesuai dengan elbow method
clus = KMeans(n_clusters= 2, n_init=10, init= 'k-means++', max_iter= 300)
clus.fit(rfm_scaled)
#Mausukka hasil cluster ke data rfm awal
rfm['K_Cluster'] = clus.labels_
rfm.head()
#Visualisasi Silhouette Analysis
visualizer = SilhouetteVisualizer(clus)

visualizer.fit(rfm_scaled) 
visualizer.poof() 
#Masukkan semua ke dalam data yang sudah di scaling 
rfm_scaled['K_Cluster'] = clus.labels_
rfm_scaled['RFM_Level'] = rfm.RFM_Level
rfm_scaled.reset_index(inplace = True)
rfm_scaled.head()
#melting data frame yang telah dibuat
rfm_melted = pd.melt(frame= rfm_scaled, id_vars= ['customer_unique_id', 'RFM_Level', 'K_Cluster'], 
                     var_name = 'Metrics', value_name = 'Value')
rfm_melted.head()
#Visualisasi snake plot
fig, ax = plt.subplots(figsize=(16, 9))
sns.lineplot(x = 'Metrics', y = 'Value', hue = 'RFM_Level', data = rfm_melted)
plt.title('Snake Plot of RFM')
plt.legend(loc = 'upper right')
#Visualisasi snake plot dengan K-Means
fig, ax = plt.subplots(figsize=(16, 9))
sns.lineplot(x = 'Metrics', y = 'Value', hue = 'K_Cluster', data = rfm_melted)
plt.title('Snake Plot of K_cluster')
plt.legend(loc = 'upper right')
#Berapaka jumlah customer berdasarkan kategorinya?
rfm_cus_level = rfm_scaled.groupby('RFM_Level')['customer_unique_id'].nunique().reset_index()

#Visualisasi
fig=plt.figure(figsize=(16,9))
sns.barplot(y=rfm_cus_level['customer_unique_id'], x=rfm_cus_level['RFM_Level'], palette="Greens_d")
plt.title('Customer Based on RFM Level',fontsize=20)
plt.xlabel('RFMLevel',fontsize=17)
plt.ylabel('Amount of Customer',fontsize=17)