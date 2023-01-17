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
my_data = pd.read_csv("/kaggle/input/world-happiness/2015.csv")
my_data.info()
my_data.head()
my_data.tail()
my_data.corr()
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(my_data.corr(), annot = True, linecolor='purple',linewidths= .5, square=True, fmt = '.1f',ax=ax)

plt.show()
my_data.columns
my_data.Family.plot(kind='line',color='blue',label='Family',linewidth=1.5,alpha=0.7,grid=True,linestyle=':')

my_data.Freedom.plot(color='darkgreen',label='Freedom',linewidth=1.5,alpha=0.7,grid=True,linestyle='-.')

plt.legend(loc = 'lower left')

plt.xlabel('x axis')

plt.ylabel('y axis') 

plt.title('Line Plot')

plt.show()
my_data.plot("Happiness Score","Family",grid=True)

my_data.plot("Happiness Score","Freedom",grid=True)

plt.show()
my_data.plot(kind='scatter', x='Freedom', y='Generosity', alpha=0.5, color='green')

plt.xlabel('Freedom') 

plt.ylabel('Generosity')

plt.title('Freedom Generosity Scatter Plot') 

plt.show()
my_data.Freedom.plot(kind="hist",bins=50,figsize=(12,12))

plt.show()
m = my_data['Family'] > 1.35

my_data[m]
my_data[(my_data['Freedom']>0.62) & (my_data['Family']>1.35)]
my_data[(my_data['Freedom']>0.62) & (my_data['Family']>1.4) & (my_data['Generosity']>0.3)]