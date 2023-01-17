# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")

data.info()
data.corr()
#correlation map

f , ax  = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".1f",ax=ax)

plt.show()
data.head(10)
#line Plot

#color =color ,label =label , lindwidth of line , alpha =opacity , grid =grid , linestyle style

#data.suicides_no.plot(kind ="line",color ="g" , label ="Suicides_no",linewidth =1,alpha = 0.5,grid = True , linestyle = "-.")

#data."suicides/100k pop".plot( color ="r" , label ="Population",linewidth =1,alpha = 0.5,grid = True , linestyle = ":")



#plt.legend(loc="upper right")

#plt.xlabel("x axis")

#plt.ylabel("y axis")

#plt.title("Line Plot")
data.columns

data.population.plot(kind="line",color="g",label="population",linewidth=1,alpha=0.5,grid= True,linestyle=":")



data.suicides_no.plot(color="r",label="suicides_no",linewidth=1,alpha=0.5,grid=True,linestyle="-.")

plt.legend(loc="upper right")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Line Plot")

plt.show()
series=data["population"]

print(series)
x=data["population"]<500

data[x]

#data[np.logical_and(data["population"]<10000,data["suicides_no"]>1)]

#data[(data["population"]<100000) & (data["suicides_no"]>1)]