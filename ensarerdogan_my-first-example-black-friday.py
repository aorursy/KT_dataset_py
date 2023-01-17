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
data=pd.read_csv("../input/BlackFriday.csv")

data.info()

data.head()
#collaration map example



f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,linewidth=5,fmt='.1f',ax=ax)

plt.show()
#line plot example

data.Purchase.plot(kind="line",color="blue",label="Purchase",linewidth=1,alpha=0.5,grid=True,linestyle=":")

data.User_ID .plot(kind="line",color="green",label="User_ID",linewidth=50,alpha=0.5,grid=True,linestyle=":")

plt.legend(loc='upper right')

plt.xlabel('x Purchase')

plt.ylabel('y User_ID')

plt.title('Line Plot')

plt.show()
#scatter plot example



data.plot(kind="scatter",x='Purchase',y='Occupation',alpha=0.005,color="red")

plt.xlabel('Purchase')

plt.ylabel('Occupation')

plt.title('scatter plot')

plt.show()
#histogram plot example



data.Purchase.plot(kind="hist",bins=10,figsize=(8,8))

plt.xlabel('Purchase')

plt.show()
#list comprehension example



avg=sum(data.Purchase)/len(data.Purchase)

print("avg :",avg)

data["Purchase_level"]=["high" if i > avg else "low" for i in data.Purchase]

data.loc[:10,["Purchase_level","Purchase","User_ID"]]