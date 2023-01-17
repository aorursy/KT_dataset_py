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
# List pada Python

a = [1, 2, 3]

b = [4, 5, 6]

a + b
# Penjumlahan dilakukan iterasi

result = []

for first, second in zip(a, b):

    result.append(first + second)

result
import numpy as np



#Membuat array

a = np.array([1, 2, 3])

a
b = np.array([1, 2, 3.0])

b
# Mengecek tipe data array

type(a)
a = np.array([1, 2, 3])

a.dtype
a = np.array([1, 2, 3], dtype='int32')

a.dtype
# Cek jumlah dimensi pada array

a = np.array([1, 2, 3])

a.ndim
# Cek shape dari array

a = np.array([1, 2, 3])

a.shape
a = np.array([1, 2, 3])

f = np.array([1.1, 2.2, 3.3])

a +f
a * f
a ** f
# Universal function (unfucs) pada nupy seperti sin

np.sin(a)
# Mengakses elemen pada indeks ke-0

a = np.array([1, 2, 3])

a[0]
# Melakukan assign pada element array dengan element baru

a[0] = 10

a
a = np.array([1, 2, 3])

a.dtype
# Assign nilai baru pada indeks ke-0

a[0] = 11.6

a
# Membentuk array 2-dimensi

a = np.array([[ 0, 1, 2, 3],

             [10, 11, 12, 13]])

a
a.shape
a.size
a.ndim
a[1, 3]
a[1, 3] = -1

a
a = np.array([1, 2, 3])

# Slicing dari indeks ke-1, dan batas indeks ke-2

a[1:2]
# Slicing dari indeks ke-1, dan batas indeks ke-1 dari be

a[1:-1]
a[::2]
import pandas as pd
# Membuat series

obj = pd.Series([1, 2, 3])

obj
# Cek tipe

type(obj)
# Membuat series dengan indeks

obj2 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

obj2
# Mendapatkan index obj2

obj2.index
# Mendapatakan values obj2

obj2.values
# Menampilkan value pada index a

obj2['a']
# Assign nilai baru pada index a

obj2['a'] = 4

obj2
# Akses nilai pada index a dan c

obj2[['a', 'c']]
# Membuat objek series baru

obj3 = pd.Series([4, 5, 6], index=['a', 'd', 'c'])

obj3
obj2 + obj3
# Membuat data frame

data = {'kota': ['Semarang', 'Semarang', 'Semarang',

                'Bandung', 'Bandung', 'Bandung'],

       'Tahun': [2016, 2017, 2018, 2016, 2017, 2018],

       'Populasi': [1.5, 2.1, 3.2, 2.3, 3.2, 4.5]}

frame = pd.DataFrame(data)

frame
# Cek tipe

type(frame)
# Cek shape dari frame

frame.shape
# Cek info dari frame

frame.info()
# Menampilkan data 5 teratas

frame.head()
# Menampilkan data 5 terbawah

frame.tail()
# Column pada dataframe

frame.columns
# Index pada dataframe

frame.index
# Mendapatkan keseluruhan data

frame.values
#Statistika pada data numeric

frame.describe()
# Statistik untuk data string

frame['kota'].value_counts()
# Akses data pada kolom a

frame['Populasi']
# Akses data pada baris ke-3, berarti indeks ke-2

frame.loc[2]
# Akses data pada indeks 2-3

frame.loc[2:3]
# Akses data pada baris ke-3, berarti indeks ke-2

frame.loc[2]
# Akses elemen pada kolom populasi index ke-2

frame['Populasi'][2]
import matplotlib.pyplot as plt

%matplotlib inline
import numpy as np



x = np.linspace(0,2*np.pi, 100)

cos_x = np.cos(x)
# Membuat line plot

fig, ax = plt.subplots()

_= ax.plot(x, cos_x)
# Membuat sumbu x dan y memiliki rasio yang sama

fig, ax =plt.subplots()

_= ax.plot(x, cos_x)

_= ax.set_aspect('equal')
fig, ax = plt.subplots()

_= ax.plot(x, cos_x, markersize=20, linestyle='-.',

          color='red', label='cos')

_= ax.set_aspect('equal')

_= ax.legend
# Membuat scatter plot

x = np.array([1, 2, 3])

y = np.array([4, 5, 6])



fig, ax = plt.subplots()

_= ax.scatter(x, y)

_= ax.set_xlabel('x axis')

_= ax.set_ylabel('y axis')
# Membuat bar plot

kategori = ['Panas', 'Dingin']

jumlah = [8, 5]



fig, ax = plt.subplots()

_= ax.bar(kategori, jumlah)

_= ax.set_xlabel('Kategori')

_= ax.set_ylabel('Jumlah')

_= ax.set_title('Penikmat Kopi')