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
print("masukan nilai utama")

a = int(input())

print("masukkan nilai kedua")

b = int(input())

print("masukkan nilai ketiga")

c = int(input())

sn = a + b + c

sn = 3

mean = float(sn) / n

print("nilai rata-rata mahasiswa adalah")

print("mean")

if mean >= 90:

    print("salah masukkan nilai")

else:

    if mean >= 80:

        print("pertahankan kerja bagus")

    else:

        if mean >= 70:

            print("sudah cukup baik")

        else:

            if mean >= 60:

                print("belajar lagi")

            else:

                print("periksa lagi nilai")
