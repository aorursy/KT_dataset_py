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
data_2015=pd.read_csv('../input/world-happiness/2015.csv')

data_2016=pd.read_csv('../input/world-happiness/2016.csv')

data_2017=pd.read_csv('../input/world-happiness/2017.csv')

data_2018=pd.read_csv('../input/world-happiness/2018.csv')

data_2019=pd.read_csv('../input/world-happiness/2019.csv')
data_2015.info()
data_2015.head()
data_2015.columns
data_2015.corr()

f,ax=plt.subplots(figsize=(10,9))

sns.heatmap(data_2015.corr(),annot=True,fmt='.2f',ax=ax,vmin=-1, vmax=1, center= 0, cmap= 'coolwarm',linewidths=3, linecolor='black')

plt.show()
f,ax=plt.subplots(figsize=(10,9))

d = data_2015.loc[lambda data_2015: data_2015['Region'] == 'Eastern Asia']

matrix = np.triu(d.corr())

sns.heatmap(d.corr(), annot=True, mask=matrix)
data_2015.plot(kind='scatter',x='Happiness Score',y='Health (Life Expectancy)',color='red',alpha=.5)

plt.xlabel('Happiness Score')

plt.ylabel('Health')

plt.title('Happiness Score vs Health')

plt.show()
data_2015.plot(kind='scatter',x='Economy (GDP per Capita)',y='Health (Life Expectancy)',color='red',alpha=.5,label="Economy vs Health")

plt.xlabel('Economy')

plt.ylabel('Health')

plt.legend(loc="center left")

plt.title('Economy vs Health')

plt.show()
data_2015['Economy (GDP per Capita)'].plot(kind = "line", color = "purple",label = "Economy (GDP per Capita)",linewidth = 1, alpha = 0.5, grid = True,figsize=(10,6), marker='s', ms=10)

data_2015['Health (Life Expectancy)'].plot( color = "red",label = "Health (Life Expectancy)",linewidth = 1, alpha = 0.5, grid = True,figsize=(12,6), marker='o', ms=10)

plt.legend(loc='upper right')

plt.xlabel('Economy')

plt.ylabel('Life Expectancy')

plt.title('Economy vs Life Expectancy, 2015')

plt.show()
data_2015[['Country','Happiness Rank']]

x=data_2015["Happiness Rank"]<5

data_2015[x]
data_2015 = data_2015["Happiness Score"].head()

data_2019 = data_2019["Score"].head()



comp_data = pd.concat([data_2015,data_2019],axis = 1)

comp_data