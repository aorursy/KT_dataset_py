# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#menginput library 

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("..//input"))



customers = pd.read_csv('..//input//Mall_Customers.csv')

#input data ke dalam notebook



customers.head()

#melihat data
customers.describe()

#Kita melihat penjelasan data secara statistik deskriptif 
customers.shape

#melihat bentuk dari data yang terdiri dari 200 baris dan 5 kolom
customers.dtypes

#melihat tipe dari data tersebut
customers.isnull().sum()

#melihat apakah terdapat null atau tidak
sns.countplot(x='Gender', data=customers)

plt.style.use('Solarize_Light2') #memilih style plot

plt.title('Distribusi Gender')

#visualisasi data dengan melihat distribusi gender

#dapat terlihat bahwa gender female atau perempuan lebih mendominasi dalam pengunjung mall
customers.hist('Age', bins=35, color = 'darkolivegreen')

plt.title('Distribusi Umur')

plt.xlabel('Age')



#Visualisasi data untuk distribusi umur dengan bins 35 dikarenakan mengambil nilai tengah 0-70

#berdasarkan grafik dapat kita lihat bahwa pengunjung mall terbanyak berumur diantara 30 - 33


plt.hist('Age', data=customers[customers['Gender'] == 'Male'], alpha=0.5, label='Laki-Laki', color = 'blue')

plt.hist('Age', data=customers[customers['Gender'] == 'Female'], alpha=0.5, label='Perempuan', color ='red')

plt.title('Distribusi Umur oleh Gender')

plt.xlabel('Umur')

plt.legend()



#Visualisasi data dengan histogram melihat distribusi umur pengunjung mall berdasarkan gendernya

#dapat dilihat bahwa perempuan dengan umur diatas 30 mendominasi sebagai pengunjung terbanyak

#sedangkan pengunjung yang sedikit adalah laki-laki dan perempuan berumur diantara 59-65 tahun


customers.hist('Annual Income (k$)')

plt.title('Distribusi Pendapatan Tahunan dalam Ribu Dollars')

plt.xlabel('Ribu Dollar')



#visualisasi data untuk distribusi pendapatan tahunan

#distribusi pendapatan terbesar berada di kisaran 60 hingga 80 ribu dollar
plt.hist('Annual Income (k$)', data=customers[customers['Gender'] == 'Male'], alpha=0.5, label='Laki-laki', color = 'green')

plt.hist('Annual Income (k$)', data=customers[customers['Gender'] == 'Female'], alpha=0.5, label='Perempuan', color = 'red')

plt.title('Distribusi Pendapatan berdasarkan Gender')

plt.xlabel('Pendapatan (Ribu Dollars)')

plt.legend()



#apakah gender mempengaruhi pendapatan? maka dari itu kita mencari distribusinya

#berdasarkan tabel di bawah dapat terlihat bahwa pendapatan lebih banyak oleh laki-laki dibandingkan perempuan

#Pendapatan perempuan terbesar berada di kisaran 60-80 ribu dollar


customers.hist('Spending Score (1-100)', color = 'darkgreen')

plt.title('Distribusi Pengeluaran dalam Skor 1-100')

plt.xlabel('Skor')



#visualisasi data untuk distribusi pengeluaran dalam skor 1 sampai 100

#distribusi pendapatan terbesar berada di kisaran skor 40-60
plt.hist('Spending Score (1-100)', data=customers[customers['Gender'] == 'Male'], alpha=0.5, label='Laki-laki', color = 'orange')

plt.hist('Spending Score (1-100)', data=customers[customers['Gender'] == 'Female'], alpha=0.5, label='Perempuan', color = 'brown')

plt.title('Distribusi Pengeluaran berdasarkan Gender')

plt.xlabel('Pengeluaran (Skor 1-100)')

plt.legend()



#apakah gender mempengaruhi pengeluaran? maka dari itu kita mencari distribusinya

#berdasarkan tabel di bawah dapat terlihat bahwa pengeluaran lebih banyak oleh perempuan dibandingkan laki-laki

#Pendapatan perempuan terbesar berada di kisaran skor 60-80 
#kita memisahkan customer perempuan dan laki-laki

male_customers = customers[customers['Gender'] == 'Male']

female_customers = customers[customers['Gender'] == 'Female']



#selanjutnya kita mencari rata-rata pengeluaran dari keduanya

print(male_customers['Spending Score (1-100)'].mean())

print(female_customers['Spending Score (1-100)'].mean())



#dapat dilihat bahwa rata-rata pengeluaran perempuan yang terbesar dibandingkan laki-laki
Seg = customers.iloc[:, [3,4]].values



#untuk melihat segmentasi, kita memisahkan data yang akan di segmenkan

#kita menggunakan data pendapatan dan pengeluaran



print(Seg)
from sklearn.cluster import KMeans

wcss = [] #menentukan nilai wcss

for n in range(1, 11): #range dengan looping 10 kali, jika 20 kali dengan range 22

    kmeans = KMeans(n_clusters = n, init = 'k-means++', random_state = 0) 

    #n_cluster adalah jumlah cluster, init adalah jumlah K yang dipilih.

    #kita menggunakan k-means++ karena menghindari jebakan centroid

    #random state adalah jika kita memilih 0 di kesempatan berbeda maka bilangan random akan sama

    kmeans.fit(Seg)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, 'o')

plt.plot(range(1, 11), wcss, '-')

plt.title('Metode Elbow')

plt.xlabel('Jumlah cluster')

plt.ylabel('WCSS')

plt.show()
#membuat model K-Means

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)

y_kmeans = kmeans.fit_predict(Seg)

#untuk unsupervised learning kita gunakan "fit_predict()" sedangkan untuk supervised learning menggunakan "fit_tranform()
print(y_kmeans)

#untuk lebih simpel kita convert ke bentuk data frame atau tabel

df_kmeans = pd.DataFrame(y_kmeans)

df_kmeans.head(10)



#array/data frame di bawah memiliki arti yaitu pengunjung 1 masuk ke cluster 5 (karena 0 adalah 1 dan 4 adalah 5) dst
# Visualisasi hasil clusters

plt.figure(1 , figsize = (17 , 8))

plt.scatter(Seg[y_kmeans == 0, 0], #sumbu x atau 0 dengan kluster 0 atau 1

            Seg[y_kmeans == 0, 1], #sumbu y atau 1 dengan kluster 0 atau 1

            s = 100, #size adalah 100

            c = 'maroon', #warna pola plot

            label = 'Kluster 1') #nama dari plot

plt.scatter(Seg[y_kmeans == 1, 0], Seg[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Kluster 2')

plt.scatter(Seg[y_kmeans == 2, 0], Seg[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Kluster 3')

plt.scatter(Seg[y_kmeans == 3, 0], Seg[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Kluster 4')

plt.scatter(Seg[y_kmeans == 4, 0], Seg[y_kmeans == 4, 1], s = 100, c = 'orange', label = 'Kluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')

plt.title('Kluster pengunjung')

plt.xlabel('Pendapatan tahunan (Ribu Dollar)')

plt.ylabel('Skor pengeluaran (1-100)')

plt.legend()

plt.show()



#visualisasi dengan lebih simpel

plt.figure(1 , figsize = (17 , 8))

plt.scatter(Seg[y_kmeans == 0, 0], Seg[y_kmeans == 0, 1], s = 100, c = 'maroon', label = 'Kaya dan Hemat')

plt.scatter(Seg[y_kmeans == 1, 0], Seg[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Seimbang')

plt.scatter(Seg[y_kmeans == 2, 0], Seg[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Kaya dan Boros')

plt.scatter(Seg[y_kmeans == 3, 0], Seg[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Sederhana dan Boros')

plt.scatter(Seg[y_kmeans == 4, 0], Seg[y_kmeans == 4, 1], s = 100, c = 'orange', label = 'Sederhana dan Hemat')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')

plt.title('Kluster pengunjung')

plt.xlabel('Pendapatan tahunan (Ribu Dollar)')

plt.ylabel('Skor pengeluaran (1-100)')

plt.legend()

plt.show()