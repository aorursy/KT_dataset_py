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
#Nama : Rizqi Fachreza

#NIM  : 20.01.53.3001



uts = [60,70,50,70,80,70,90,80,40,75]

uas = [70,80,60,90,70,75,90,70,60,85]

i=0

Lulus=0

TidakLulus=0

for i in range (10):

    nilaiakhir=(uts[i]*0.4)+(uas[i]*0.6)

    if(nilaiakhir>70):

        print("Nilai akhir:" ,nilaiakhir,"Mahasiswa Lulus")

        Lulus+=1

    else:

        print("Nilai akhir:" ,nilaiakhir,"Mahasiswa Tidak Lulus")

        TidakLulus+=1



print("Mahasiswa yang Lulus : ",Lulus)

print("Mahasiswa yang Tidak Lulus : ",TidakLulus)