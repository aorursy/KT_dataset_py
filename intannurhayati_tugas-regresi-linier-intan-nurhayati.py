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
m1 = [0] * (9)

m2 = [0] * (6)

hasil = [0] * (6)



jumlah = 0

index = 0

for x in range(0, len(m1) - 1 + 1, 1):

    m1[x] = x

print("m1= ", end='', flush=True)

for x in range(0, len(m1) - 1 + 1, 1):

    print(m1[x], end='', flush=True)

print("")

for x in range(0, len(m2) - 1 + 1, 1):

    m2[x] = x

print("m2= ", end='', flush=True)

for x in range(0, len(m2) - 1 + 1, 1):

    print(m2[x], end='', flush=True)

for x in range(0, 2 + 1, 1):

    for y in range(0, 1 + 1, 1):

        for z in range(0, 2 + 1, 1):

            jumlah = jumlah + m1[z + (x + 3)] * m2[z * 2 + y]

        hasil[index] = jumlah

        jumlah = 0

        index = index + 1

print("")

print(" Hasil perkalian dua matriks ")

for x in range(0, len(hasil) - 1 + 1, 1):

    print(str(hasil[x]) + ",", end='', flush=True)
