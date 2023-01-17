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
data = pd.read_csv("../input/creditcard.csv")
data.info()
data.corr()
data.head(9)
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt = '.1f', ax=ax, linecolor='b')
plt.show()
data.columns
data.Time.plot(kind='line', color = 'red', label= 'Time', linewidth = 1, alpha = 0.5, grid=True, linestyle=':')
data.Amount.plot(kind='line', color = 'yellow', label= 'Amaount', linewidth = 1, alpha = 0.5, grid=True, linestyle='-')
plt.legend(loc='upper left')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line plot')
plt.show()
data.plot(kind='scatter',x='Time', y='Amount',alpha= 0.5, color = 'red', edgecolor='green')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.show()
data.V18.plot(kind='hist', bins=50, figsize=(15,15))
plt.show()
series = data['Amount']
df = data[['Amount']]
#print(series)
#print(df)
x = data['Amount'] > 10000
data[x]

data[(data['Time'] > 200000) | (data['Amount'] > 12000)]
for index,value in data[['Amount']][0:1].iterrows():
    print(index, ' : ', value)
