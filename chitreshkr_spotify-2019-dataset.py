# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #matplot library

plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding = "ISO-8859-1")
df.head()
df.tail()
df.columns
df.shape
df.info()
df.describe()
df['Track.Name'].describe()
df['Track.Name'].unique()
df['Artist.Name'].describe()
df['Artist.Name'].unique()
df['Genre'].describe()
df['Genre'].unique()
df.plot(y='Popularity',x= 'Track.Name',kind='bar',figsize=(25,6),legend =True,title="Popularity Vs Track Name",

        fontsize=18,stacked=True,color=['r', 'g', 'b', 'r', 'g', 'b', 'r'])

plt.ylabel('Popularity')

plt.xlabel('Track Name')

plt.show()
df.plot(y='Beats.Per.Minute', kind='box')
df.plot(kind='box',subplots=True, layout=(4,3),figsize=(35,20),color='r',fontsize=22,legend = True)
df[df['Popularity'] == df['Popularity'].max()] 
df[df['Popularity'] == df['Popularity'].min()] 
df['Popularity'].nlargest(5)
df[df['Popularity'] == 94]
df[df['Popularity'] == 93]
df[df['Popularity'] == 92]