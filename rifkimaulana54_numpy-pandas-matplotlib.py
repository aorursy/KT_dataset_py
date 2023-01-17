Nim  = 41190037
Nama = 'Muhamad Rifki Maulana' 
Matkul = 'Machine Learning'
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
print("hello world")
import numpy as np 
a = np.array([1, 2, 3])
b = a[2] 
b
a = np.array([1, 2, 3])
a.ndim
a = np.array([1, 2, 3])
a.shape
#array multidimensi
import numpy as np 
buah = np.array([
    ["mangga", "jeruk" "apel"],
    ["anggur", "pisang", "semangka"]
])
print(buah[0])
print(buah[1][2])
import pandas as pd
obj = pd.Series([1, 2, 3])
obj
#membuat dataframe 
data = {'kota' : ['Semarang', 'Semarang', 'Semarang',
                 'Bandung', 'Bandung', 'Bandung'], 
        'tahun' : [2016, 2017, 2018, 2016, 2017, 2018], 
        'populasi' : [2.5, 1.5, 1.7, 3.5, 2.2, 1.5]} 
frame = pd.DataFrame(data)
frame
frame.shape
frame.info()
#menampilkan 5 data teratas
frame.head()
#menampilkan 5 data dari bawah
frame.tail()
 
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
x = np.linspace(0,2*np.pi, 100)
cos_x = np.cos(x) 
fig, ax = plt.subplots() 
_= ax.plot(x, cos_x)
fig, ax = plt.subplots() 
_= ax.plot(x, cos_x) 
_= ax.set_aspect('equal')
kategori = ['Panas', 'Dingin'] 
jumlah = [8, 2]

fig, ax = plt.subplots()
_= ax.bar(kategori, jumlah) 
_= ax.set_xlabel('kategori')
_= ax.set_ylabel('jumlah')
_= ax.set_title("Penikmat Kopi")