# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sn

import warnings





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/daimonds1/diamonds.csv")
df.head()
df.price
[1,2,3]
df.info()
df.price.hist(bins=20)
df.price.hist()
n,bins,_=plt.hist(df.price)
n=np.array([25335.,  9328.,  7393.,  3878.,  2364.,  1745.,  1306.,  1002., 863.,   726.])

n
bins
plt.hist(df.price,bins=20)
plt.hist(df.price,bins=30)
sn.distplot(df.price)
sn.distplot(df.price,hist=False)
#shows the mean price for each color

sn.barplot(x=df.color,y=df.price)
from numpy import median

sn.barplot(x=df.color,y=df.price,data=df,estimator= median)

l=[50,201,236,269,271,278,283,291,301,380,490]
np.percentile(l,25),np.percentile(l,50),np.percentile(l,75)
#Q3-Q1=IQR

(296-252.5)
#Q1-1.5*IQR,   Q3+1.5*IQR ---- Cutoffs for outliers

252.5-1.5*43.5,296+1.5*43.5




sn.set_style("whitegrid")

sn.boxplot(l)



box=plt.boxplot(df.price)
box




#Min and Max values of the distribution

[item.get_ydata()[0] for item in box['caps']]







#Q1 and Q3 is 25 and 75 percentile

[item.get_ydata()[0] for item in box['whiskers']]







#Median

[item.get_ydata()[0] for item in box['medians']]



[df.price.median()]
df['price']
#give out

df.price[df.price>11883].count()
sn.boxplot(y='price',x='cut',data=df)
sn.boxplot(y='price',x='clarity',data=df)
sn.boxplot(x='price',y='clarity',data=df)
plt.scatter(df.carat,df.price) 
data={'Year':['2001','2002','2003','2004','2005','2006','2007','2008'],'Salary':[1000,1600,1200,1300,1100,1300,1800,1200]}

dff=pd.DataFrame(data)

dff
sn.lineplot(x=dff.Year,y=dff.Salary,data=dff)
ax=sn.heatmap(df[['carat','depth','price']].corr(),annot=True)
df.columns
sn.pairplot(df[['carat','depth', 'table', 'price']])