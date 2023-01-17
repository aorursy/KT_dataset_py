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
def RA(data):

    jumlah=0

    for i in range (len(data)):

        jumlah+=data[i]

        rata=jumlah/len(data)

    print("jumlah data:",len(data))

    print("Total nilainya adalah: ",jumlah)

    print("Nilai rata-ratanya adalah: ",rata)



data = ([60,70,90,65,75,55,70,55,85,990,660])

RA(data)

def SD(data):

    jumlah=0

    for i in range (len(data)):

        jumlah +=data[i]

        

    rata=jumlah/len(data)

    sigma = 0

    for i in range (len(data)):

        hitung = (data[i]-rata)**2

        sigma += hitung

        pembagianN=sigma/len(data)

        standarDeviasi=pembagianN **0.5

        print("Nilai Standar Deviasi adalah:",standarDeviasi)

        

data =([60,70,90,65,75,55,70,55,85,990,660])

SD(data)