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
data = pd.read_csv('../input/creditcardfraud/creditcard.csv') #read the file like type csv

print(data)
data.info() #show to information about data .

print(data.columns) 

data.describe()
f , ax =plt.subplots()

sns.heatmap(data.corr() , annot=True , linewidths= 4 ,  fmt = '.1f' , ax=ax)

#plt.clf() if ı want to clean the data , ı can use the this method.

plt.show()

data.V3.plot(kind='line' , color='green' , label='V3' , linewidth=0.5 , grid=True , linestyle=':')

data.V4.plot(color='red', label='V4' , linewidth=0.5 , grid=True , linestyle=':')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('compile between amount to green')

plt.legend(loc='upper right')

plt.show()
print(data['Amount']," - " ,data['Class'])
data.plot(kind = 'line' , x='V3' , y='V4' , alpha=0.5 , color='red'  )

plt.xlabel('V3')

plt.ylabel('V4')

plt.title('compile between amount to green')

plt.legend(loc='upper right')

plt.show()
data.Class.plot(kind='hist' , bins=50)

plt.title("V4")

plt.legend(loc='upper right')

plt.show()
x=data['V3'] > 1.5

print(x)
print(data[np.logical_and(data['V3'] > 1.5 , data['V4']>1)])