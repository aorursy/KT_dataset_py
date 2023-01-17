# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/german-credit/german_credit_data.csv')

data
data.info()
# summary statistic

data.describe()
# count_missing = data.isnull().sum().sort_values(ascending = False)

percentage_missing=round(data.isnull().sum()/len(data)*100,2).sort_values(ascending = False)

print(percentage_missing)
df=data.drop(columns=['Unnamed: 0'])

df.head()
# fill the missing value

categoricals = ['Checking account','Saving accounts']



for num in categoricals :

    modus=df[num].mode().values[0]

    df[num] = df[num].fillna(modus)



df.head()
# boxplot weather



f=plt.figure(figsize=(13,12))

f.add_subplot(3,3,1)

sns.boxplot(df['Credit amount'],orient = "h")

f.add_subplot(3,3,2)

sns.boxplot(df['Duration'],orient = "h")

f.add_subplot(3,3,3)

sns.boxplot(df['Age'],orient = "h")



# ketiga variabel tidak dihapus, karena akan digunakan untuk analisis
df.head()
df['Purpose'].unique()
f=plt.figure(figsize=(30,30))

f.add_subplot(3,3,1)

sns.countplot(df['Sex'])

f.add_subplot(3,3,2)

sns.countplot(df['Housing'])

f.add_subplot(3,3,3)

sns.countplot(df['Saving accounts'])

f.add_subplot(3,3,4)

sns.countplot(df['Checking account'])

f.add_subplot(3,3,5)

sns.countplot(df['Purpose'])



# terlihat bahwa kreditur didominasi oleh laki-laki,dengan kepemilikan rumah sendiri, dengan tujuan meminjam untuk kredit mobil.

# selain itu, pada grafik dapat diketahui bahwa dominasi kepemilikan tabungan kreditur berada pada kategori little atau sedikit,

# begitu pun dengan checking account atau pengeluaran sehari-hari dari akun bank, masuk dalam kategori little atau sedikit.
age=df['Age'].unique()

age
plt.figure(figsize=(15,5))

ages=df[df['Age']>18]

sns.countplot(ages['Age'], order = ages['Age'].value_counts(), hue=ages['Sex'])

plt.title('Demografi Kreditur Berdasarkan Umur dan Gender')

plt.show()



# Terlihat bahwa kreditur didominasi oleh umur 21 - 35 tahun.
df['Duration'].value_counts()
plt.figure(figsize=(24,10))

ax3 = sns.distplot(df['Duration'])

ax3.set(title="Durasi Peminjaman Kredit")



# Pada grafik terlihat bahwa durasi peminjaman kredit terbanyak berada pada rentang tidak lebih dari 24 bulan,

# dimana setelah 24 bulan tren grafik menunjukkan penurunan, yang artinya tidak banyak kreditur yang meminjam

# dengan durasi lebih dari 24 bulan.
# mean CREDIT AMOUNT

credit=df[['Saving accounts', 'Credit amount']]

total=credit.groupby('Saving accounts').mean().sort_values('Credit amount',ascending=False).head()

total
plt.figure(figsize=(7,3))



x=range(4)

plt.bar(x,total['Credit amount'])

plt.xticks(x,total.index)

plt.xlabel('Saving account')

plt.ylabel('Credit amount')

plt.title('Credit amount of Saving account category')

plt.show()



# Terlihat bahwa jumlah kredit yang paling banyak didominasi oleh kreditur yang memiliki saving account kategori menengah ke bawah,

# Hal ini dapat diambil insight untuk memberikan lebih banyak promosi kepada nasabah yang memiliki saving account dengan

# kategori menengah ke atas, agar semakin banyak lagi yang melakukan kredit.
plt.figure(figsize=(15,5))

plt.title('Purpose Kredit Berdasarkan Saving Account Category')

sns.countplot(x="Saving accounts", hue='Purpose', data=df)



# Terlihat bahwa mayoritas purpose kredit adalah car atau mobil untuk semua saving account kategori.
# scatterplot Horse Power vs Price

plt.scatter(df['Credit amount'],df['Duration'])

plt.title('Credit amount vs Duration')

plt.xlabel('Duration')

plt.ylabel('Credit amount')

plt.show()



# terlihat bahwa, semakin pendek durasinya maka jumlah kredit semakin rendah.
#create correlation with hitmap



#create correlation

corr = df.corr(method = 'spearman')



#convert correlation to numpy array

mask = np.array(corr)



#to mask the repetitive value for each pair

mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots(figsize = (15,12))

fig.set_size_inches(20,10)

cmap=sns.cm.rocket_r

sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True, cmap=cmap)



# terlihat bahwa, besar credit ammount banyak dipengaruhi oleh durasi peminjaman dan pekerjaan.