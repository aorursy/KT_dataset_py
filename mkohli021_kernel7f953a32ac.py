# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
d=pd.read_csv('/home/mahima/Desktop/bitcoinData.csv')

d.head()
d.tail()
d.shape

d.shape
d.count()
d.describe()
d['Date']=pd.to_datetime(d['Date'])
d.dtypes
data=d.set_index("Date")

data.head()
d.corr()
data['avg']=(data['Open*']+data['Close**'])/2

data.head()
x=data['Open*']

y=data['Close**']

plt.figure(figsize=(15,8))

plt.plot(x,color='r')

plt.plot(y,color='b')

plt.show()
plt.figure(figsize=(15,8))

plt.plot(data.index,data.avg)

plt.title('data')