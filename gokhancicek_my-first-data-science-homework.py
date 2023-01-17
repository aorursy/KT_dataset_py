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
data = pd.read_csv('../input/clinvar_conflicting.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()
data.head(10)
data.tail(10)
data.columns
# Line Plot
data.CADD_PHRED.plot(kind = 'line', color = 'blue', label = 'CADD_PHRED', linewidth = 1, Alpha = .7, grid = True, linestyle = '--' )
data.CADD_RAW.plot(color = 'red', label = 'CADD_RAW', linewidth = 1, Alpha = .4, grid = True, linestyle = ':')
plt.legend(loc = 'upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('My Line Plot')
plt.show()
# Scatter Plot
# x =  CADD_PHRED, y = CADD_RAW
data.plot(kind = 'scatter', x= 'CADD_PHRED', y = 'CADD_RAW', alpha = .5, color = 'green')
plt.xlabel('CADD_PHRED')
plt.ylabel('CADD_RAW')
plt.title('My Scatter Plot')
plt.show()
# Histogram
data.CADD_RAW.plot(kind = 'hist', bins = 50, figsize = (10,10))
plt.show()
dictionary = {'galatasaray':'hagi','fenerbahce':'alex','besiktas':'sergen'}
print(dictionary.keys())
print(dictionary.values())
dictionary['fenerbahce'] = "lugano"
print(dictionary)
dictionary['trabzonspor'] = "yattara"
print(dictionary)
del dictionary['besiktas']
print(dictionary)
print('galatasaray' in dictionary)
dictionary.clear()
print(dictionary)
series = data['Feature']
print(type(series))
data_frame = data[['Feature']]
print(type(data_frame))
# Comparison operators
print(2 == 2)
print(3>=2)
print(21<=15)

# Boolean operators
print(True and False)
print(True or False)
print(False and False)
print(False or True)
# Filtering pandas data frame
x = data['CADD_PHRED']>80.0
data[x]
data[(data['CADD_PHRED']>90) | (data['CADD_RAW']<-5)]
# While and For Loops
i = 0
while i != 7:
    print('i is:',i)
    i +=1
print('my favorite number is 7')
liste = [7,14,21,28,35,42,49]
for i in liste:
    print('ab is:', i)
print('')
for index, value in enumerate(liste):
    print(index," : ",value)
print('')
dictionary = {'galatasaray': 'kewell','milan':'kaka'}
for key, value in dictionary.items():
    print(key,":",value)
    





