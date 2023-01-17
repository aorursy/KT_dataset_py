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
#Buatlah program untuk menampilkan deret Fibonacci menggunakan perulangan while.
totalprev = 0
bil1 = 0
bil2 = 1
billoop = 1
total = 10
while billoop <= total:
        print('Deret Fibonacci =')
        print(totalprev)
        totalprev = bil1 + bil2
        bil2 = bil1
        bil1 = totalprev
        billoop = billoop + 1
#Buatlah program untuk menghitung factorial.
bilangan = int(input('Masukkan Bilangan = '))
bilulang = 1
hasil = 1
while bilulang <= bilangan:
        hasil = hasil * bilulang
        bilulang = bilulang + 1
print('Hasil Faktorialnya adalah =',hasil)