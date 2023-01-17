# list pada pyhton
a = [1,2,3]
b = [4,5,6]
a+b
# penjumlahan
result = []
for first, second in zip(a, b):
    result.append(first + second)
result
import numpy as np
# membuat array
a = ([1, 2, 3])
a
b = np.array([1, 2, 3.0])
b
# membuat type data array
type(a)
a = np.array([1, 2, 3])
a.dtype
a = np.array([1, 2, 3], dtype='int64')
a.dtype
# cek shape dari array
a = np.array([1, 2, 3])
a.ndim
# cek shape
a = np.array([1, 2, 3])
a.shape
# oprasi array
a = np.array([1, 2, 3])
f = np.array([1.1, 2.2, 3.3])
a + f
a * f
a ** f
# universal function
np.sin(a)
# mengakses elemen pada indeks
a = np.array([1, 2, 3])
a[0] = 10 
a
# melakukan asign pada elemnt array
a[0] = 10
a
a[0] = 11.6
a
# membentuk arrat 2-dimensi
a = np.array([[0, 1, 2, 3],
             [10, 11, 12, 13]])
a
a.shape
a.size
a.ndim
a[1, 3]
a[1, 3] = -1
a
a = np.array([1, 2, 3])
# slicing dari indeks
a[1:2]
a[1:-1]
a[::2]
import pandas as pd
# membuat series 
obj = pd.Series([1, 2, 3])
obj
# CEK
type(obj)
# membuat series dengan indeks
obj2 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
obj2
obj2.index
obj2.values
obj2['a']
obj2['a'] = 4
obj2
obj2[['a', 'c']]
# membuat objek series baru
obj3 = pd.Series([4, 5, 6], index=['a', 'b', 'c'])
obj3
obj2 + obj3
# membuat dataframe 
data = {'kota': ['semarang', 'semarang', 'semarang',
                'bandung', 'bandung', 'bandung'],
                'tahun': [2016, 2017, 2018, 2016, 2017, 2018],
                'populasi': [1.5, 2.1, 3.2, 2.3, 3.2, 4.5]}
frame = pd.DataFrame(data)
frame
type(frame)
# cek shape dari frame
frame.shape
frame.info()
# menampilkan data 5 teratas
frame.head()
# 5 terbawah
frame.tail()
# colom pada dataframe
frame.columns
# index pada dataframe
frame.index
# mendapatkan keseluruhan data
frame.values
# statistika pada data numeric
frame.describe()
# stastik untuk data string
frame['kota'].value_counts()
# akses data pada kolom a
frame['populasi']
# akses data pada baris 3
frame.loc[2]
# akses data pada indek
frame.loc[2:3]
# akses data pada baris ke 3, index ke 2
frame.loc[2]
# data pada index 2 - 3
frame.loc[2:3]
frame['populasi'][2]
# import libary
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

x = np.linspace(0.2*np.pi, 100)
cos_x = np.cos(x)
# membuat line plot
fig, ax = plt.subplots()
_ = ax.plot(x, cos_x)
# membuat sumbu x dan y memiliki rasio yang sama
fig, ax = plt.subplots()
_ = ax.plot(x, cos_x)
_ = ax.set_aspect('equal')
fig, ax = plt.subplots()
_ = ax.plot(x, cos_x, markersize=20, linestyle='-.', 
            color='red', label='cos')
_ = ax.set_aspect('equal')
_ = ax.legend()
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

fig, ax = plt.subplots()
_ = ax.scatter(x, y)
_ = ax.set_xlabel('x axis')
_ = ax.set_ylabel('y axis')
# membuat bar plot 
kategori = ['panas', 'dingin']
jumlah = [8, 5]

fig, ax = plt.subplots()
_ = ax.bar(kategori, jumlah)
_ = ax.set_xlabel('kategori')
_ = ax.set_ylabel('jumlah')
_ = ax.set_title('penikmat senja')



