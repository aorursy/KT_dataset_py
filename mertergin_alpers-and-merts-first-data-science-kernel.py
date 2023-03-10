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
data = pd.read_csv('/kaggle/input/cryptocurrency-financial-data/consolidated_coin_data.csv') 
data

data = data[:2000]
data
#String data tiplerini floata çevrildi. (İçiçe for la floata çeviremedik.) 

for i in range(0,data.Open.size):

    data.Open[i] = float(data.Open[i])

for i in range(0,data.High.size):

    data.Open[i] = float(data.High[i])

for i in range(0,data.Low.size):

    data.Open[i] = float(data.Low[i])

for i in range(0,data.Close.size):

    data.Open[i] = float(data.Close[i])

    

    
for i in data.Open:

    print(type(i))
#import matplotlib.pyplot as plt

plot =data.Open.plot(kind='line')

#plt.show()
