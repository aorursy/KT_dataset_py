# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head(10)


data.info()


f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()


data.corr()
data.reviews_per_month.plot(kind = 'line', color = 'g',label = 'reviews_per_month',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.price.plot(color = 'r',label = 'price',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')          

plt.ylabel('y axis')

plt.title('Line Plot')            

plt.show()
data.plot(kind='scatter', x='reviews_per_month', y='price',alpha = 0.5,color = 'red')

plt.xlabel('reviews_per_month')            

plt.ylabel('price')

plt.title('reviews_per_month priceScatter Plot')   
data.availability_365.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.xlabel('availability_365')

plt.show()
series = data['availability_365']       

print(type(series))

data_frame = data[['availability_365']] 

print(type(data_frame))
x = data['availability_365']>300

data[x]
x = data['availability_365']==0

data[x]
x = data['availability_365']>0

data[x]
x = data['availability_365']>-1

data[x]
data[np.logical_and(data['availability_365']>200, data['price']>150 )]
data[(data['availability_365']>300) & (data['price']>150)]
for index,value in data[['price']][0:100].iterrows():

    print(index," : ",value)