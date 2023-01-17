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
#List pada python

a =[1,2,3]

b =[4,5,6]

a+b
#Penjumlahan dilakukan literasi

result = []

for first, second in zip(a, b):

    result.append(first + second)

    result
#mengimpor file numpy

import numpy as np
#membuat array

a = np.array([1, 2, 3])

a
#membuat array 2 tipe upcasting

b = np.array([1, 2, 3.0])

b
#mengecek tipe data pada array

type(a)
#mengecek tipe data pada element array

a = np.array([1, 2, 3])

a.dtype
#menetapkan tipe data dengan menggunakan parameter dtype

a = np.array([1, 2, 3], dtype='int32')

a.dtype
#mengecek jumlah dimensi pada array

a = np.array([1, 2, 3, 4])

a.ndim
#mengecek shape dari array berikut

a = np.array([1, 2, 3, 4 , 5, 6])

a.shape
#operasi penjumlahan pada array

a = np.array([1, 2, 3])

c = np.array([1.2, 2.5, 3.7])

a + c
#operasi perkalian pada array

a * c
#operasi perkalian*perkalian array

a ** c
#universal function (ufuncs) pada numpy seperti sin

np.sin(a)
#mengakses elemen pada indeks ke-2

a = np.array([1, 2, 3, 4, 5, 6])

a[2]



#kenapa hasil indeks array tersebut 3? karena array dimulai dari 0 ganteng
#melakukan operasi assignment, apa itu assign? setau gua memberi nilai ke suatu variabel

#contoh misalnya nilai a kita kasih 25, jadi penulisan nya a = 25 

a[0] = 25 

a
a = np.array([1, 2, 3])

a.dtype
#assign nilai baru pada indeks-0

a[0] = 11.6

a



#kenapa hasil nilai tersebut 11? bukan 11.6?, karena tipe data tersebut int bukan float
#membentuk array 2 dimensi

a = np.array([[0, 1, 2, 3],

             [10, 11, 12, 13]])

a
#fungsi shape pada array menghasilkan jumlah baris dan jumlah kolom

a.shape
#mengetahui jumlah element pada array multi.dimensi

# 2 x 4 = 8 

a.size
#mengetahui jumlah dimensi array

a.ndim
#mengetahui nilai elemen pada indeks array

#mendapatkan data pada baris 1 dan kolom 3

a[1,3]
#mengubah data pada array dengan nilai baru (operasi assignment)

a[1, 3] = 80

a
a = np.array([1, 2, 3, 4, 5, 6])

#slicing dari indeks -1 dan batas indeks ke-4

a[1:4]
a = np.array([1, 2, 3, 4, 5, 6])



a[1:-1]
#menggunakan library pandas (python for data analysis)

import pandas as pd
#membuat Series

obj = pd.Series([10,20,30,40])

obj



#Secara default, pandas secara otomatis memberi index pada setiap baris dari series 
type(obj)
#membuat atau mendefinisikan index pada Series

#membust Series dengan index

obj2 = pd.Series([10, 30, 60, 90, 120], index=['A', 'B', 'C', 'D', 'E'])

obj2
#mendapatkan index pada obj2

obj2.index
#mendapatkan values pada obj2

obj2.values
#menampilkan atau mengambil nilai 'value' pada indeks A

obj2['A']
#menambahkan nilai baru (assign) pada indeks A

obj2['A'] = 1000

obj2
#mengakses nilai pada indeks a dan c

obj2[['A', 'C']]

#membuat objek Series baru

obj3 = pd.Series([40, 50, 60, 70, 80], index=['A', 'B', 'C', 'D', 'E'])

obj3
#operasi aritmatika penambahan pada Series

obj2 + obj3
#DataFrame

data = {'kota': ['Bandung', 'Bandung', 'Bandung', 'Jakarta', 'Jakarta', 'Jakarta'],

        'tahun': [2018, 2019, 2020, 2018, 2019, 2020],

        'popluasi': [1.5, 2.1, 3.2, 2.3, 3.2, 4.5]}

frame = pd.DataFrame(data)

frame
#cek tipe

type(frame)
#cek shape dari frame

frame.shape
#cek info dari frame

frame.info()
#menampilkan data 5 teratas

frame.head()
#menampilkan data 5 terbawah

frame.tail()
#menampilkan kolom pada object

frame.columns
#menampilkan index pada dataframe

frame.index
#menampilkan kesuluruhan data

frame.values
#statistika pada data numeric

frame.describe()
#statistik untuk data string

frame['kota'].value_counts()
#mengakses data pada kolom populasi

frame['popluasi']
#mengakes data pada baris ke-3 berarti indeks -2

frame.loc[2]
#akses data pada indeks 2-3

frame.loc[2:3]
#akses elemen pada kolom populasi indeks ke -2

frame['popluasi'][2]
#import library

import matplotlib.pyplot as plt

%matplotlib inline
import numpy as np



x = np.linspace(0,2*np.pi,100)

cos_x = np.cos(x)
#membuat line plot

fig, ax = plt.subplots()

_ = ax.plot(x, cos_x)
#membuat sumbu x dan y memiliki rasio yang sama

fig, ax = plt.subplots()

_ = ax.plot(x, cos_x)

_ = ax.set_aspect('equal')
#membuat scatter plot

x = np.array([1, 2, 3])

y = np.array([4, 5, 6])



fig, ax = plt.subplots()

_ = ax.scatter(x, y)

_ = ax.set_xlabel('x axis')

_ = ax.set_ylabel('y axis')
#membuat bar plot

kategori = ['Panas', 'Dingin']

jumlah = [8, 5]



fig, ax = plt.subplots()

_ = ax.bar(kategori, jumlah)

_ = ax.set_xlabel('kategori')

_ = ax.set_ylabel('jumlah')

_ = ax.set_title('penikmat kopi')