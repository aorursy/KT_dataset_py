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
df = pd.read_csv('/kaggle/input/esport-earnings/E-Sport Earnings.csv')
df.head()
# Data visualization

df['Earnings'] = df['Earnings'].astype('int64')
df['Genre'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns

# Let us count the amount of money in each sport

f,ax = plt.subplots(1,1,figsize=(18,8))

x = df['Earnings'].groupby(df['Genre']).sum()

ax.pie(df['Earnings'].groupby(df['Genre']).sum() , labels = x.index,autopct = '%1.1f%%',shadow = True)

plt.show()
# data visualisation

f,(ax1,ax2) = plt.subplots(1,2,figsize=(18,8))

lst = ['Fighting','FPS','Sports','Stratergy','Collectable Card','BR','TPS']

explode = (0.1,0.1,0.1,0.1,0.1,0.1,0.1)

ax1.pie(df['Genre'].value_counts(),labels = lst,autopct= '%1.1f%%',explode = explode,shadow = True)

ax1.set_title('Genre')

ax2.bar(lst,df['Genre'].value_counts(),color='r')

plt.show()
# Turning the Tournaments held into a label column to use some unnessary mechine learning

# Lets preprocess the data