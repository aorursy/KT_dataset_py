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
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()
data.info()
data.corr() #correlation values.
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax) #correlation değerlerinin doğru-ters orantılı oduğunun tablosal görünümü.
plt.show()
data.head()
data.columns
data.age.plot(kind = 'line',figsize=(10,10), color = 'blue',label = 'age',linewidth=2,alpha = 0.8,grid = True,linestyle = ':')
data.trestbps.plot(color = 'red',label = 'trestbps',linewidth=2, alpha = 0.9,grid = True,linestyle = '-')
plt.legend(loc='upper right')     #figürün boyutu,değerlerin görünümünün ayarlandığı line tablosu.
plt.ylabel('y axis')
plt.title('Age and Trestbps Table')            # title = title of plot
plt.show()
data.plot(kind='scatter',figsize=(8,8), x='oldpeak', y='age',alpha = 0.8,color = 'red')
plt.xlabel('Oldpeak')             
plt.ylabel('Age')
plt.title('Sex and Target Scatter Plot')  
series = pd.Series(4 * np.random.rand(4),
                       index=['trestbps', 'target', 'oldpeak', 'chol'], name='series') #değerlerin dağılımı.
series.plot.pie(figsize=(10, 10))
data.chol.plot(kind = 'hist',bins = 20,color='red',figsize = (10,10))

plt.show()
x = data['chol']>400  #kolestrolü 400 den büyük olanlar.
data[x]
data[np.logical_and(data['chol']>300, data['age']>50 )]
data[np.logical_and(data['chol']>300, data['age']>50 )].sex.plot(kind = 'hist',bins = 20,color='red',figsize = (10,10))

plt.show() #kolestrolü 300 den ve yaşı 50den büyük insanların cinsiyete göre dağılımı.
data.boxplot(column='chol', by='age',figsize=(15,15))
plt.show()