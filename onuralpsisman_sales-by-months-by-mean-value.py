# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/salesbymonths/satislar.csv')
data.shape
#Checking for missing data

data.columns

#"Aylar" means months and "Satislar" means sales
#Checking for missing data

data.info()
data.corr()
#Finding mean value for sales

Mean=data.Satislar.mean()
data.Satislar.plot(kind='line',color='Blue',Alpha=0.5,figsize=(9,7),label='Sales')

plt.plot([0,35],[Mean,Mean],label='Average',color='Red')

plt.legend()

plt.xlabel('Months')

plt.ylabel('Sales')

plt.grid(True)
 #Looking for the first year

data12=data.head(12)
Mean12=data12.Satislar.mean()
data12.Satislar.plot(kind='line',color='Blue',Alpha=0.5,figsize=(9,7),label='Sales')

plt.plot([0,35],[Mean12,Mean12],label='Average',color='Red')

plt.legend()

plt.xlabel('Months')

plt.ylabel('Sales')

plt.grid(True)
fig = plt.figure(figsize = (18,6))

sns.barplot(x = 'Aylar', y = 'Satislar', data = data)

plt.xlabel('Months')

plt.ylabel('Sales')

plt.plot([0,24],[Mean,Mean],label='Average',color='Red')

plt.legend()

plt.grid(True)

plt.title("Sales by months")

plt.show()
data1=data[data.Satislar>data.Satislar.mean()]

data2=data[data.Satislar<data.Satislar.mean()]

data2.Satislar=(data2.Satislar-data.Satislar.mean())

data1.Satislar=(data1.Satislar-data.Satislar.mean())

#Making "zero point" as average value
data3=pd.concat([data1,data2])

#combinin datas
fig = plt.figure(figsize = (18,6))

sns.barplot(x = 'Aylar', y = 'Satislar', data = data3)

plt.xlabel('Months')

plt.ylabel('Sales')

plt.grid(True)

plt.title("Sales by months according to mean value")

plt.show()