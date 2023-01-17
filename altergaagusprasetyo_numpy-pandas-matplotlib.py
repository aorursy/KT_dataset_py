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
#list pada python
a = [1,2,3]
b = [4,5,6]
a + b
#penjumlahan
result = []
for first, second in zip(a,b):
    result.append(first+second)
result
import numpy as np
#membuat array
a=np.array([1,2,3])
a
b=np.array([1,2,3.0])
b
#mengecek tipe data array
type (a)
a=np.array([1,2,3])
a.dtype
a=np.array([1,2,3], dtype='int64')
a.dtype
#cek jumlah dimensi array
a=np.array([1,2,3])
a.ndim
#cek shape dari array
a=np.array([1,2,3])
a.shape
a=np.array([1,2,3])
f=np.array([1.1,2.2,3.3])
a+f
a*f
a**f
#universal function pada numpy
np.sin(a)
#mengakses elemen
a=np.array([1,2,3])
a[0]
#melakukan assign pada elemen
a[0]=0
a
a=np.array([1,2,3])
a.dtype
#assign nilai baru
a[0]=11.6
a
#membnetuk array 2 dimensi
a=np.array([[0,1,2,3],
           [10,11,12,13]])
a
a.shape
a.size
a.ndim
a[1,3]
a[1,3]=-1
a
a=np.array([1,2,3])
#slicing dari indeks
a[1:2]
#slicing dari indeks ke-1
a[1:-1]
a[::2]
import pandas as pd
#membuat series
obj=pd.Series([1,2,3])
obj
#cek tipe
type(obj)
#membuat series dengan index
obj2=pd.Series([1,2,3], index=['a', 'b', 'c'])
obj2
#mendapatkan index obj2
obj2.index
#mendapatkan value obj2
obj2.values
#menampilkan value index a
obj2['a']
#assign nilai baru index a
obj2['a']=4
obj2
#akses nilai pada index a dan c
obj2[['a','c']]
#membuat objek series baru
obj3=pd.Series([4,5,6], index=['a','d','c'])
obj3
obj2+obj3
#membuat data frame
data={'kota': ['semarang','semarang','semarang',
              'bandung','bandung','bandung'],
     'tahun': [2016, 2017,2018,2016,2017,2018],
     'populasi': [1.5,2.1,3.2,2.3,3.2,4.5,]}
frame=pd.DataFrame(data)
frame
#cek tipe
type(frame)
#cek shape dari frame
frame.shape
#cek info dari frame
frame.info()
#menampilkan data 5 teratas
frame.head()
#menampilakn data 5 terbawah
frame.tail()
#column pada dataframe
frame.columns
#index data frame
frame.index
#mendapatkan seluruh data
frame.values
#statiska pada data numeric
frame.describe()
#statiska untuk data string
frame['kota'].value_counts()
#akses data pada kolom a
frame['populasi']
#akses data pada kolom a
frame['populasi']
#akses data pada baris ke3 index ke2
frame.loc[2]
#akses data pada indeks 2-3
frame.loc[2:3]
#akses data pada baris ke3 index ke2
frame.loc[2]
#akses data pada index 2-3
frame.loc[2:3]
#akses elemen pada kolom populasi index ke2
frame['populasi'][2]
#import library
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
x=np.linspace(0,2*np.pi,100)
cos_x=np.cos(x)
#membuat line plot
fig,ax=plt.subplots()
_=ax.plot(x,cos_x)
#membuat sumbu x dan y memiliki rasio sama
fig,ax=plt.subplots()
_=ax.plot(x, cos_x)
_=ax.set_aspect('equal')
fig, ax = plt.subplots()
_ = ax.plot(x, cos_x, markersize=20, linestyle='_.',
            color='red', label='cos')
_ = ax.set_aspect('equal')
_ = ax.legend()
#membuat scatter plot
x = np.array([1,2,3])
y = np.array([4,5,6])

fig, ax = plt.subplots()
_ = ax.scatter(x, y)
_ = ax.set_xlabel('x axis')
_ = ax.set_ylabel('y axis')
#membuat bar plot
kategori = ['panas','dingin']
jumlah = [8,5]

fig, ax = plt.subplots()
_= ax.bar(kategori, jumlah)
_= ax.set_xlabel('kategori')
_= ax.set_ylabel('jumlah')
_= ax.set_title('penikmat kopi')