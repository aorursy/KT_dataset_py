# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
df.info()
df.corr()
f,ax =plt.subplots(figsize=(18,18))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

df.columns
#plot

df.oldpeak.plot(kind='line', color='g', label ='oldpeak',linewidth=1, alpha=0.5,grid=True,linestyle=':')

df.age.plot(color='r', label ='age',linewidth=1, alpha=0.5,grid=True,linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
# x=age , y=target

df.plot(kind='scatter', x='age', y='target', alpha=0.1, color='g',grid=True)

plt.xlabel('age')

plt.ylabel('target')



plt.show()
#Histogram

#bins = number of bar in figure

df.age.plot(kind='hist', bins=50,figsize=(12,12),grid=True)

plt.show()
# Filtering pandas data frame

x=df['age']>40   

df[x]
x=(df['age']>40)& (df['target']== 1)  

df[x]
male=df['sex'] == 1

df[male]
data=(df['age']<=55) & (df['age']>=50) & (df['sex'] == 0) 

df[data]