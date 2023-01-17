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
#Program Python Menghitung Nilai Mahasiswa
print ("Program Python Menghitung Nilai Mahasiswa")
print ("Nilai Minimal Kelulusan 70")
i=0
uts = [60,70,50,70,80,70,90,80,40,75]
uas = [70,80,60,90,70,75,90,70,60,85]
nilaimin = 70
lulus = 0

for i in range (10): 
    hasil = 0.4*uts[i]+0.6*uas[i]
    
    if hasil >=nilaimin:
        lulus += 1
    
    print("Nilai UTS dan UAS Mahasiswa = ", uts[i]," dan ",uas[i], " nilai akhir mahasiswa = ",hasil , "Dinyatakan Lulus.")
else :
            
    print("Nilai UTS dan UAS Mahasiswa = ", uts[i]," dan ",uas[i], " nilai akhir mahasiswa = ",hasil , "Dinyatakan Tidak Lulus.")
    
    print("Jumlah Mahasiswa Yang Lulus: " , lulus)
    print("Jumlah Mahasiswa Yang tidak Lulus: " , (10) - lulus)
    
