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
df = pd.read_csv("/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")

df.head()
df.info()
df.corr()
import matplotlib.pyplot as plt

import seaborn as sns

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
df.head(8)
df.columns

df.High.plot(kind = 'line', color = 'blue',label = 'High',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df.Close.plot(color = 'red',label = 'Close',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')            

plt.show()
df.plot(kind='scatter', x='Open', y='Close',alpha = 0.5,color = 'red')

plt.xlabel('Open')           

plt.ylabel('Close')

plt.title('Attack Open Close Plot')
df.High.plot(kind = 'hist',bins = 50,figsize = (8,8))

plt.show()
df.High.plot(kind = 'hist',bins = 50)

plt.clf()
dictionary = {'turkey' : 'istanbul','usa' : 'newyork'}

print(dictionary.keys())

print(dictionary.values())
dictionary['turkey'] = "kocaeli"    

print(dictionary)

dictionary['italy'] = "roma"       

print(dictionary)

del dictionary['turkey']              

print(dictionary)

print('italy' in dictionary)        

dictionary.clear()                   

print(dictionary)
series = df['Low']        

print(type(series))

data_frame = df[['Low']]  

print(type(data_frame))
print(3 > 2)

print(3!=2)



print(True and False)

print(True or False)
x = df['High']>15000     

data[x]
df[np.logical_and(df['Low']>2500, df['Close']>1000 )]
for index,value in df[['High']][0:1].iterrows():

    print(index," : ",value)