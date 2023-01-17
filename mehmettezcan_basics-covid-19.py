# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv')

data.head()
data.info()
#Delete unusefull column



del data['Unnamed: 0']
data.columns
data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
f,ax = plt.subplots(figsize=(18, 5))

data.Confirmed.plot(kind = 'line', color = 'g',label = 'Confirmed',linewidth=2,alpha = 1,grid = True,linestyle = ':', ax=ax)

data.Deaths.plot(color = 'r',label = 'Deaths',linewidth=2, alpha = 1,grid = True,linestyle = '-.', ax=ax)

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Relation between Confirmed and Deaths')            # title = title of plot

plt.show()
f,ax = plt.subplots(figsize=(18, 5))

data.plot(kind='scatter', x='Confirmed', y='Recovered',alpha = 0.5,color = 'red', ax=ax)

plt.xlabel('Confirmed')              # label = name of label

plt.ylabel('Recovered')

plt.title('Confirmed Recovered Scatter Plot')

plt.show()
f,ax = plt.subplots(figsize=(18, 5))

data.Suspected.plot(kind = 'hist',bins = 50,figsize = (12,12),ax=ax)

plt.show()