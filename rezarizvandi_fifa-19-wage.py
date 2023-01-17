# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import re

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/fifa19/data.csv')

data.head()
data.drop(columns='Unnamed: 0' ,inplace=True)

data.head()
data.columns
df = data[['Name','Nationality','Club','Wage']]

df.head()
df.shape
df.drop_duplicates(inplace = True)

df.dropna(inplace = True)
for i in range(len(df)):

    df.iloc[i]['Wage'] = df.iloc[i]['Wage'][1:-1]

df['Wage'] = df['Wage'].replace('',0)
df.reset_index(inplace=True)
df.drop(columns='index' , inplace = True)
count = 0

i = 0

for item in df.values :

    if 100 < int(item[-1]) < 500:

        count += 1

        print(item[0])

    else : 

        df.drop(index= i,inplace = True)

    i += 1

print('The Number of Players who get payed between 100K and 500K Euro is',count)
df.sort_values('Wage',ascending=False)
plt.figure(figsize = (16,8))

plt.title('Most paied players by nationality')

sns.set_style('darkgrid')

#plt.subplot(1,1,1)

sns.countplot(df['Nationality'])

plt.xticks(rotation = 60)

plt.show()

plt.figure(figsize = (16,8))

plt.title('Most paied Players by Club')

sns.countplot(df['Club'])

plt.xticks(rotation = 90)

plt.show()