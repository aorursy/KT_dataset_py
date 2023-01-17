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
#tentukan jumlah deret fibonacci

Angka = 10

 

# tentukan angka pertama

n1 = 0

# tentukan angka kedua

n2 = 1

# jumlah angka yang dihitung

count = 2

 

# periksa angka

if Angka <= 0:

   print("Angka harus di atas 0")

elif Angka == 1:

   print("Deret fibonacci : ",Angka,":")

   print(n1)

else:

   print("Deret fibonacci : ",Angka,":")

   print(n1,",",n2,end=', ')

   while count < Angka:

       nth = n1 + n2

       print(nth,end=' , ')

       # tukar nilai untuk mendapatkan 2 index terakhir

       n1 = n2

       n2 = nth

       count += 1
# fungsi rekursif untuk perhitungan faktorial



def faktorial(n):

   if n == 1:

       return n

   else:

       return n*faktorial(n-1)



# tentukan angka untuk dicari nilai faktorialnya

num = 4



# cek angka

if num < 0:

   print("Angka harus di atas 0")

elif num == 0:

   print("Faktorial dari 0 adalah 1")

else:

   print("Faktorial dari ",num," adalah ",faktorial(num))