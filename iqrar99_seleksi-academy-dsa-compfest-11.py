# EDA oleh Tim Haha Iya Bang



import numpy as np # aljabar linear

import pandas as pd # data processing

import matplotlib.pyplot as plt # visualisasi data

import seaborn as sns # visualisasi data

%matplotlib inline



# Input data

data_lokasi = pd.read_csv("../input/catatan_lokasi.csv")

profil_karyawan = pd.read_csv("../input/data_profil.csv") 
print(data_lokasi.head(10))

print("ukuran data: "+ str(data_lokasi.shape))
data_lokasi.iloc[:, 2].describe()
# ada 28 lokasi yang berbeda, oleh karena itu kami penasaran dengan lokasi-lokasi tersebut

data_lokasi.iloc[:, 2].unique()
data_lokasi.info()
print(profil_karyawan.head(10))

print("ukuran data: "+ str(profil_karyawan.shape))
print(profil_karyawan['umur'].describe())

print("Median Umur : "+str(profil_karyawan['umur'].median()))

print("----------------------------------------")

print(profil_karyawan.iloc[: , 1:3].describe())
profil_karyawan.info()
sns.set(style="darkgrid")

f, ax = plt.subplots(1,1,figsize=(18,8))

sns.countplot(x = 'divisi', data = profil_karyawan,hue = 'jenis_kelamin', palette = "Set2")

plt.show()
f, ax = plt.subplots(2,1,figsize=(15,12))

sns.violinplot(x = "divisi", y = "umur", data = profil_karyawan, split = True, scale = "count", ax = ax[0])

sns.violinplot(x = "divisi", y = "umur", data = profil_karyawan, split = True, 

               hue = "jenis_kelamin", scale = "count", palette = "Set2", ax = ax[1])



plt.show()
divisi = profil_karyawan.iloc[:,2].unique()

for d in divisi:

    tertua = max(profil_karyawan.iloc[:, 3][profil_karyawan.divisi == d])

    termuda = min(profil_karyawan.iloc[:, 3][profil_karyawan.divisi == d])

    

    print("Divisi {:25}: Tertua = {} Tahun, Termuda = {} Tahun".format(d, tertua, termuda))
# mengambil urutan tanggal

tanggal =  []

for t in data_lokasi["tanggal"]:

    if t not in tanggal: tanggal.append(t)



jabodetabek = ['Jakarta Pusat', 'Kabupaten Bogor', 'Kota Tangerang', 'Kota Bogor', 

               'Kota Depok', 'Jakarta Timur', 'Kota Bekasi', 'Jakarta Selatan', 

               'Jakarta Utara', 'Jakarta Barat', 'Kabupaten Tangerang', 'Kabupaten Bekasi', 

               'Kota Tangerang Selatan']



karyawanLuarKota = []

karyawanDalamKota = []



for t in tanggal:

    # menghitung karyawan yang pergi ke luar kota maupun tidak

    cntLK = cntDK = 0 

    

    for lokasi in data_lokasi.lokasi_dominan[data_lokasi["tanggal"] == t]:

        if lokasi in jabodetabek: cntDK += 1

        else: cntLK += 1

    

    karyawanLuarKota.append(cntLK)

    karyawanDalamKota.append(cntDK)



# menggabungkan beberapa DataFrame

df1 = pd.DataFrame(karyawanLuarKota, columns = ['Jumlah Di Luar Kota'])

df2 = pd.DataFrame(karyawanDalamKota, columns = ['Jumlah Di Dalam Kota'])

df3 = pd.DataFrame(tanggal, columns = ['Tanggal'])



data_jumlah = pd.merge(df1, df2, left_index = True, right_index= True)

data_jumlah = pd.merge(df3, data_jumlah, left_index = True, right_index= True)

print(data_jumlah)

print("------------------------------------------------------------")

print(data_jumlah.describe())
fig, ax = plt.subplots(1,1, figsize= (18,7))

ax.plot(data_jumlah["Tanggal"], data_jumlah.iloc[:,1], label = 'Di Luar Kota')

ax.plot(data_jumlah["Tanggal"], data_jumlah.iloc[:,2], label = 'Di Dalam Kota')

plt.xticks(rotation=90)

plt.legend(loc='best')



ax.set(xlabel='Tanggal', ylabel='Jumlah Orang',

       title='Grafik Lokasi Keberadaan Karyawan TokoLapak')



plt.show()
lokasi = sorted(['Jakarta Pusat', 'Kabupaten Bogor', 'Kota Tangerang', 'Kota Bogor',

       'Kota Depok', 'Jakarta Timur', 'Kota Bekasi', 'Jakarta Selatan',

       'Jakarta Utara', 'Jakarta Barat', 'Kabupaten Tangerang',

       'Kabupaten Bekasi', 'Kota Tangerang Selatan', 'Kota Yogyakarta',

       'Kabupaten Bantul', 'Kabupaten Wonogiri', 'Kota Medan',

       'Kota Manado', 'Kota Bandung', 'Kabupaten Tasikmalaya',

       'Kota Padang', 'Kabupaten Sleman', 'Kota Bandar Lampung',

       'Kota Semarang', 'Kabupaten Lampung Selatan', 'Kota Surabaya',

       'Kabupaten Kebumen', 'Kota Banda Aceh'])



jumlah_kunjungan = []

for l in lokasi:

    jumlah_kunjungan.append(len(data_lokasi.lokasi_dominan[data_lokasi.lokasi_dominan == l]))

    

df1 = pd.DataFrame(lokasi, columns = ["Lokasi"])

df2 = pd.DataFrame(jumlah_kunjungan, columns = ["Total Kunjungan"])



data_kunjungan = pd.merge(df1, df2, left_index= True, right_index = True)

data_kunjungan = data_kunjungan.sort_values("Total Kunjungan", ascending = False)

print(data_kunjungan)

print("-------------------------------------------------")

print(data_kunjungan.describe())
f, ax = plt.subplots(1,1,figsize=(18,8))

ax = sns.barplot(x = 'Lokasi', y = "Total Kunjungan" ,data = data_kunjungan, palette = "Set1")

ax.set_title('Grafik Total Kunjungan Lokasi', size = 17)

plt.xticks(rotation=90)

plt.show()
from collections import Counter



# mengambil data dari tanggal 31 Mei - 8 Juni 2019

data_mudik = data_lokasi.iloc[1000:1900 , :]

lokasi_mudik = []



for ID in range(1,101):

    lokasi_karyawan = list(data_mudik.lokasi_dominan[data_mudik.id == ID])

    modus = Counter(lokasi_karyawan).most_common(1)

    

    lokasi_mudik.append(modus[0][0])

    

ID = range(1,101)



df = list(zip(ID, lokasi_mudik))

data_lokasi_mudik = pd.DataFrame(df, columns = ["ID", "Lokasi Mudik"])

data_lokasi_mudik
jabodetabek = ['Jakarta Pusat', 'Kabupaten Bogor', 'Kota Tangerang', 'Kota Bogor', 

               'Kota Depok', 'Jakarta Timur', 'Kota Bekasi', 'Jakarta Selatan', 

               'Jakarta Utara', 'Jakarta Barat', 'Kabupaten Tangerang', 'Kabupaten Bekasi', 

               'Kota Tangerang Selatan']



persentase = [0.0,0.0] # kiri untuk yang di luar Jabodetabek, kanan sebaliknya



for l in data_lokasi_mudik["Lokasi Mudik"]:

    if l in jabodetabek: 

        persentase[1] += 1

    else:

        persentase[0] += 1



labels = "Mudik", "Tidak Mudik"



fig, ax = plt.subplots(1,1, figsize = (10, 8))

patches, texts, autotexts = ax.pie(persentase, explode=(0.1,0.1), labels = labels, 

                                   autopct='%1.1f%%', startangle=240, colors = ["gold", "yellowgreen"])

texts[0].set_fontsize(20)

texts[1].set_fontsize(20)

ax.set_title("Persentase Karyawan yang Mudik", size = 20)

ax.axis('equal')



plt.show()
# Kita akan hilangkan data lokasi yang berada di kawasan Jabodetabek agar mendapatkan data Non-Jabodetabek

jabodetabek = ['Jakarta Pusat', 'Kabupaten Bogor', 'Kota Tangerang', 'Kota Bogor', 

               'Kota Depok', 'Jakarta Timur', 'Kota Bekasi', 'Jakarta Selatan', 

               'Jakarta Utara', 'Jakarta Barat', 'Kabupaten Tangerang', 'Kabupaten Bekasi', 

               'Kota Tangerang Selatan']



df = data_lokasi_mudik

lokasi = list(data_lokasi_mudik.iloc[:, 1])



for l in list(data_lokasi_mudik.iloc[:, 1]):

    if l in jabodetabek:

        df = df.drop(df.index[lokasi.index(l)])

        lokasi.pop(lokasi.index(l))

        

f, ax = plt.subplots(1,1,figsize=(18,8))

sns.countplot(x = 'Lokasi Mudik', data = df, palette = "Set1")

plt.xticks(rotation = 90)

ax.set_title("Daerah Asal Karyawan TokoLapak", size = 17)

plt.show()
# mengambil urutan tanggal

tanggal =  []

for t in data_lokasi["tanggal"]:

    if t not in tanggal: tanggal.append(t)

        

total_pemudik = []

for t in tanggal:

    cnt = 0

    hari = 1

    df = list(data_lokasi.lokasi_dominan[data_lokasi["tanggal"] == t])

    for n in range(1,101):

        if df[n-1] == list(data_lokasi_mudik["Lokasi Mudik"])[n-1]:

            cnt +=1

    total_pemudik.append(cnt)

    hari += 1

    

df1 = pd.DataFrame(tanggal, columns = ['Tanggal'])

df2 = pd.DataFrame(total_pemudik, columns = ['Total Pemudik'])

pemudik = pd.merge(df1,df2,right_index= True, left_index = True)



print(pemudik)