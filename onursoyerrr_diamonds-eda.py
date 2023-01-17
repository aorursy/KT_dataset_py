# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading data

data = pd.read_csv("/kaggle/input/diamonds/diamonds.csv")

data.head()
data.info()
data.corr()
#correlation map 

f,ax = plt.subplots(figsize = (13,13))

sns.heatmap(data.corr(),annot = True , linewidth = 5, fmt = '.1f' , ax = ax)

plt.show()

#visualising the correlations between columns 
#line plot 

data.price.plot(kind = 'line' , label = 'Price' , grid = True, alpha = 0.5 , linewidth = 1 , linestyle = '-.',color = 'r')

data.carat.plot(label = 'Carat', grid = True , alpha = 0.5 , linewidth = 1 , linestyle = '-.', color = 'g')

plt.legend(loc = 'upper right')

plt.xlabel('Price')                    

plt.ylabel('Carat')

plt.title('Line Plot')

plt.show()
data.isnull().values.any()

#checking if there is any missing value
data_num = data.select_dtypes(include = ['int64','float64'] )

data_num.head()

#selection of numerical values in data to observing statistical values
data_num.describe().T

#observing statistical values in data
#scatter plot

data.plot(kind = 'scatter' , x = 'price' , y = 'carat', color = 'r' , alpha = 0.5)

plt.xlabel("price")

plt.ylabel("carat")

plt.show()

#histogram 

data.plot(kind = 'hist' , bins = 50, figsize = (13,13))

plt.show()
#plt.clf()

data.plot(kind = 'hist' , bins = 50, figsize = (13,13))

plt.clf()
x = data['price']>18000

data[x]
data[(data['price']>1800) & (data['table']>90)]