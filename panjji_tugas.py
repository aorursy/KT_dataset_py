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
#masukan semua variable
uts = [60, 70, 50, 70, 80, 70, 90, 80, 40, 75]
uas = [70, 80, 60, 90, 70, 75, 90, 70, 60, 85]
NA = 70
L = 0
TL= 0

#statement nilai minimal kelulusan 
print ("nilai minimal kelulusan " + str(NA))

#perulangan sejumlah data / Lenght dari salah satu variable, UAS atau UTS
for i in range(len(uas)):

#Rumus
    rata = 0.4*uts[i]+0.6*uas[i]
    if rata >= NA:
#cetak mahasiswa L atau TL
        print(rata)
        L=L+1
    else:
        TL=TL+1
#hasil
print("JML MHS LULUS: ")
print(L)
print("JML MHS TDK LULUS: ")
print(TL)