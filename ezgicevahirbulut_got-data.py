# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/game-of-thrones/battles.csv')
print(data)
result=data.head()
print(result)

result=data.tail()
print(result)
result=data["name"].head()
print(result)
data.info()
result=data[["name","year"]].head()
print(result)
result=data[["name","year"]].tail(7)
print(result)

result=data[5:][["name","year"]].head()
print(result)
data.info()

reasult=data[data["year"]>=200][["name","year"]].head(25)
print(result)
#correlation map
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f',ax=ax)
plt.show()
data.head(10)
data.columns
#line plot
data.defender_size.plot(kind='line', color='g', label='defender_size',linewidth=1, alpha=0.5,grid=True,linestyle=':')
data.attacker_size.plot(color='r',label='attacker_size',linewidth=1,alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='upper right') 
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('line plot')
plt.show()
#scatter plot
data.plot(kind='scatter', x='attacker_size', y='defender_size',alpha=1,color='red')
plt.xlabel('attack')
plt.ylabel('defence')
plt.title('attack defense scatter plot')
plt.show()
#histogram
data.defender_size.plot(kind='hist',bins=50,color='purple' ,figsize=(12,12))
plt.show()
i=0;
while i!=5:
    print('i is:',i)
    i+=1
print(i,'is equal to 5')
lis=[1,2,3,4,5]
for i in lis:
    print('i is :',i)
print('')

for index,value in enumerate(lis):
    print(index,":",value)
print('')

dictionary={'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key,":",value)
print('')
    
