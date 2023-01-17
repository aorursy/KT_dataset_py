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
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
df.info()
df.isnull().sum()
#Overview
for i in range(df.shape[1]):
    tt_rows = df[df.columns[i]].count()
    unique_rows = df[df.columns[i]].nunique()
    null_rows = df[df.columns[i]].isnull().sum()
    print(f'[unique:{unique_rows}   rows:{tt_rows}   NA:{null_rows}]------------({i}){df.columns[i]}')
#Columns with [ unique values <1000 ] & [No null value]
for i in range(df.shape[1]):
    tt_rows = df[df.columns[i]].count()
    unique_rows = df[df.columns[i]].nunique()
    null_rows = df[df.columns[i]].isnull().sum()
    if (unique_rows < 1000) & (null_rows == 0):
        print(f'[unique:{unique_rows}   rows:{tt_rows}   NA:{null_rows}]------------({i}){df.columns[i]}')
#ข้อมูลทั้งหมดแบ่งตาม 5 Location
for i in df[df.columns[4]].unique():
    print(i)
#ข้อมูลทั้งหมดแบ่งตาม 3 room_type
for i in df[df.columns[8]].unique():
    print(i)
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.subplots(figsize = (12,5))
sns.countplot(x = 'room_type', hue = 'neighbourhood_group', data = df)
#ดูปริมาณห้องโดยแบ่งตาม Location และ Area
df.groupby(['neighbourhood_group','neighbourhood'])['id'].count()
#Price
print(f"Average of price per night : ${df.price.mean():.2f}")
print(f"Maximum price per night : ${df.price.max()}")
print(f"Minimum price per night that not zero : ${df[df.price != 0].price.min()}")
#Availability_365
print(f"Average availability : {df.availability_365.mean():.2f}")
print(f"Maximum availability : {df.availability_365.max()}")
print(f"Minimum availability : {df.availability_365.min()}")
#เทียบระหว่างราคาห้องกับวันที่ว่างให้จอง
plt.figure(figsize=(12, 8))
plt.scatter(df.price, df.availability_365  , cmap='summer', edgecolor='black', linewidth=1, alpha=0.75)
plt.figure(figsize=(12, 8))
plt.scatter(df.price, df.availability_365  , cmap='summer', edgecolor='black', linewidth=1, alpha=0.75)
plt.axvline(x=1000, color='r', linestyle='--')
mprice_df = df[(df.price <= 20000) & (df.price != 0)]
mprice_df['price_range']=pd.qcut(mprice_df['price'],5)
mprice_df.groupby(['price_range'])['id'].count().to_frame()
unique_rows = df[df.columns[15]].nunique()
print(f'availability_365 มี unique value ทั้งหมด : {unique_rows}')
#dt['Fare_Range']=0
#dt.loc[dt['Fare']<=7.91,'Fare_Range']=0
#dt.loc[(dt['Fare']>7.91)&(dt['Fare']<=14.454),'Fare_Range']=1
#dt.loc[(dt['Fare']>14.454)&(dt['Fare']<=31),'Fare_Range']=2
#dt.loc[(dt['Fare']>31)&(dt['Fare']<=513),'Fare_Range']=3
plt.figure(figsize=(12, 8))
plt.scatter(mprice_df.price, mprice_df.availability_365  , cmap='summer', edgecolor='black', linewidth=1, alpha=0.75)
plt.axvline(x=1000, color='r', linestyle='--')
#เตรียมข้อมูลเพื่อจัดกลุ่ม
new_date = mprice_df[['availability_365','price']]
new_date['price_range']=0
max_price = df.price.max()
initial_step = 20
bins = 5000
increment_step = round((max_price/bins)-1)
price = initial_step
for i in range(bins):
    if i == 0:
        new_date.loc[new_date['price']<=initial_step,'price_range']=i
    elif i == bins-1:
        new_date.loc[(new_date['price']>price),'price_range']=i
    else:
        new_date.loc[(new_date['price']>price)&(new_date['price']<=price + increment_step),'price_range']=i
    price = price + increment_step
    #new_date.loc[(new_date['price']>30)&(new_date['price']<=100),'price_range']=1
    #new_date.loc[(new_date['price']>100)&(new_date['price']<=200),'price_range']=2
    #new_date.loc[(new_date['price']>200)&(new_date['price']<=310),'price_range']=3
    #new_date.loc[(new_date['price']>310)&(new_date['price']<=500),'price_range']=4
    #new_date.loc[(new_date['price']>500)&(new_date['price']<=800),'price_range']=5
final_data = new_date[['price_range','availability_365']] 
final_data = final_data.groupby('price_range').agg({'availability_365':'mean'})
dataset = final_data
dataset
new_date.price_range.nunique()
dataset
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler


fig = plt.figure(figsize=(30,27))
dendogram = sch.dendrogram(sch.linkage(dataset,method='ward'),leaf_rotation=90, leaf_font_size=12,labels=dataset.index) 
plt.title("Dendrograms")  
plt.show()
fig = plt.figure(figsize=(30,27))
dendogram = sch.dendrogram(sch.linkage(dataset,method='ward'),leaf_rotation=90, leaf_font_size=12,labels=dataset.index) 
plt.axhline(y=2000, color='r', linestyle='--')
plt.title("Dendrograms")  
plt.show()
hc = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(dataset)
print(y_hc)
    
fig = plt.figure(figsize=(20,18))
plt.scatter(dataset.index,dataset['availability_365'], c=y_hc) 
plt.title('K = 2')
plt.xlabel('Price range [ID]')
plt.xticks(rotation=90)
plt.ylabel('availability_365')
plt.show()