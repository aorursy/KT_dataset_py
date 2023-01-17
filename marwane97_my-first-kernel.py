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
#read the data

data = pd.read_csv("../input/cryptocurrencypricehistory/ethereum_dataset.csv")

data.info()
#get the first 10 rows of data

data.head(10)
#coralation 

data_corr = data.corr()

data_corr
#Coralation Visualization

f,ax = plt.subplots(figsize=(16, 16))

sns.heatmap(data_corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
#columns of data

data_columns = data.columns

data_columns
#Series

a = data['eth_etherprice']

print(type(a))

print(a)
#DataFrame

b = data[['eth_etherprice']]

print(type(b)) 

print(b)
# plot as line

data.eth_gasused.plot(kind='line',color = 'b',label = 'eth_gasused',linewidth=1,alpha = 0.7,grid = True,linestyle = '-.')



data.eth_gasprice.plot(color = 'r',label = 'eth_gasprice',linewidth=1,alpha = 0.7,grid = True,linestyle = ':')



plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # xlabel = name of x label

plt.ylabel('y axis')              # ylabel = name of y label

plt.title('Line Plot')            # title = title of plot

plt.show()
#plot as scatter

plt.scatter(data.eth_etherprice,data.eth_gaslimit,alpha = 0.4,color = 'red')



plt.xlabel('eth_etherprice')              # xlabel = name of x label

plt.ylabel('eth_gaslimit')              # ylabel = name of y label

plt.title('eth_etherprice and eth_gaslimit Scatter Plot')            # title = title of plot

plt.show()
#plot as histogram

# bins = number of bar in figure

data.eth_gasused.plot(kind = 'hist',bins = 50,figsize = (15,15))

plt.title('Histogram of eth_gasused')

plt.show()
#plot as histogram

# bins = number of bar in figure

data.eth_ens_register.plot(kind = 'hist',bins = 20,figsize = (12,12),normed=True)

plt.title("Histogram of eth_ens_register")

plt.show()
a = data.eth_address < 12000

data_true = data[a]

data_true
a = (data.eth_address >= 12000) & (data.eth_address < 17000)

data_true = data[a]

data_true
data[np.logical_and(data['eth_address']>200, data['eth_tx']<4000 )].head()