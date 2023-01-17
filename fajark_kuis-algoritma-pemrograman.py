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
i = 0
nama = [""] * (10)
uts = [0] * (10)
uas = [0] * (10)
totalnilai = [0] * (10)

l = 0
tl = 0
for i in range(0, 9 + 1, 1):
    print("Masukkan nama ")
    nama[i] = input()
    print("Masukkan nilai UTS ", end='', flush=True)
    uts[i] = int(input())
    print("Masukkan nilai UAS ", end='', flush=True)
    uas[i] = int(input())
    totalnilai[i] = 0.4 * uts[i] + 0.6 * uas[i]
    print("Total nilai akhir =", end='', flush=True)
    print(totalnilai[i])
    if totalnilai[i] >= 70:
        print("Mahasiswa Lulus")
        l = l + 1
    else:
        print("Mahasiswa Tidak Lulus")
        tl = tl + 1
print("Jumlah Mahasiswa Lulus = " + str(l))
print("Jumlah Mahasiswa Tidak Lulus = " + str(tl))
