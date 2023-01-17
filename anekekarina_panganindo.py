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
df_pangan = pd.read_csv("/kaggle/input/hargapanganindonesia/pangan_19_20.csv")


df_pangan.tail(49)
print(df_pangan.shape)
df_pangan.describe()
df_pangan.isnull().sum()
df_pangan.columns

#disajikan beberapa entittas variabel pd kolom dataset tsb
new = df_pangan['Date'].str.split("/",n=2,expand=True)



df_pangan["Tanggal"] = new[0]



df_pangan["Bulan"] = new[1]



df_pangan["Tahun"] = new[2]



df_pangan.drop(columns=["Date"],inplace=True)



df_pangan.drop(columns=["Tanggal"],inplace=True)



df_pangan["Bulan"] = df_pangan["Bulan"].replace(['01'],'13')



df_pangan
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))

plt.title('Harga Beras dan Daging Umum pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Beras'],color='blue',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()


plt.figure(figsize=(15,10))

plt.title('Harga Beras Kualitas Bawah I pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Beras Kualitas Bawah I (kg)'],color='red',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()


plt.figure(figsize=(15,10))

plt.title('Harga Beras Kualitas Bawah II pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Beras Kualitas Bawah II (kg)'],color='orange',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()


plt.figure(figsize=(15,10))

plt.title('Harga Beras Kualitas Medium I pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Beras Kualitas Medium I (kg)'],color='skyblue',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()


plt.figure(figsize=(15,10))

plt.title('Harga Beras Kualitas Medium II pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Beras Kualitas Medium II (kg)'],color='orange',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()


plt.figure(figsize=(15,10))

plt.title('Harga Beras Kualitas Super I pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Beras Kualitas Super I (kg)'],color='brown',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()


plt.figure(figsize=(15,10))

plt.title('Harga Beras Kualitas Super II pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Beras Kualitas Super II (kg)'],color='purple',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()
plt.figure(figsize=(15,10))

plt.title('Harga Daging Ayam pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Daging Ayam'],color='black',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()
plt.figure(figsize=(15,10))

plt.title('Harga Daging Ayam Ras Segar pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Daging Ayam Ras Segar (kg)'],color='blue',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()
plt.figure(figsize=(15,10))

plt.title('Harga Daging Sapi pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Daging Sapi'],color='brown',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()
plt.figure(figsize=(15,10))

plt.title('Harga Daging Sapi Kualitas 1 pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Daging Sapi Kualitas 1 (kg)'],color='red',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()
plt.figure(figsize=(15,10))

plt.title('Harga Daging Sapi Kualitas 2 pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Daging Sapi Kualitas 2 (kg)'],color='red',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()
plt.figure(figsize=(15,10))

plt.title('Harga Telur Ayam pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Telur Ayam'],color='navy',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()
plt.figure(figsize=(15,10))

plt.title('Harga Telur Ayam Ras Segar pada bulan September 2019- Januari 2020')

sns.set(style="ticks")

sns.lineplot(x=df_pangan['Bulan'],y=df_pangan['Telur Ayam Ras Segar (kg)'],color='orange',linewidth=2.5, marker='o')

plt.xlabel("Bulan")

plt.ylabel("Harga")

plt.legend()

plt.show()