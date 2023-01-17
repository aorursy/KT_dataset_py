# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls","../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/creditcard.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt = '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
data.V1.plot(kind = 'line', color = 'g', label = 'V1',linewidth=1,alpha = 0.5,grid=True, linestyle = ':')



data.V28.plot(color = 'r',label = 'V28', linewidth=1, alpha = 0.5,grid =True,linestyle = '-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Credit Cards')

plt.show()
data.plot(kind='scatter',x = 'V1', y = 'V28', alpha = 0.5, color = 'red')

plt.xlabel('V1')

plt.ylabel('V28')

plt.title('first and last')
data.V1.plot(kind='hist',bins=50,figsize=(12,12))

plt.show()
data.V1.plot(kind='hist',bins = 50)

plt.clf()
dictionary = {'V1' : 'MİN', 'V28' : 'MAX'}

print(dictionary.keys())

print(dictionary.values())
dictionary['V14'] = "Middle"
print(dictionary)
data = pd.read_csv('../input/creditcard.csv')
mini = data['V1']

print(type(mini))

data_frame = data[['V1']]

print(type(data_frame))
i = data['V1']>0.5

data[i]
for index,value in data[['V28']][0:10].iterrows():

    print(index, " : ", value)