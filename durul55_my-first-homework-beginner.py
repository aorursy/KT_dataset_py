# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2015.csv')
data.info()
data.head(10)
data.corr()
# correlation map

f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(),annot=True,linewidths=.6,fmt='.1f',ax=ax)
data.columns
data.describe()
data.plot(kind='hist',x='Freedom',y='Happiness Score',color='green',linewidth=1,)

plt.xlabel('Freedom')

plt.ylabel('Happiness Score')

plt.title('scatter plot')

plt.show()