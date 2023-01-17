# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from statistics import mean



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

    



# Any results you write to the current directory are saved as output.
data2=pd.read_csv("../input/PercentagePeopleBelowPovertyLevel.csv",encoding="windows-1252")

print(data2)
print(data2.info())
print(data2.head(10))

print(data2.tail(10))
numbers=list(range(1,29330))

data2["numbers"]=numbers

data2=data2.set_index("numbers")

print(data2.index)
data2.iloc[:,2].replace("-",0.0,inplace=True)

data2.iloc[:,2]=data2.iloc[:,2].astype(float)

data_list=list(data2["Geographic Area"].unique())

data_rate=[]

for i in data_list:

   x= data2[data2["Geographic Area"]==i]

   data_rate.append(mean(x.poverty_rate))
data3=pd.DataFrame({"States":data_list,"Values":data_rate})

plt.figure(figsize=(20,20))

ax=sns.barplot(data3.iloc[:,0],data3.iloc[:,1],palette=sns.cubehelix_palette(len(data3.iloc[:,0])))

plt.xlabel("States")

plt.ylabel("Values")

plt.show()
#pie plotting as optional

fig,ax= plt.subplots(figsize=(15,15))

ax.pie(data3.Values,  labels=data_list, autopct='%1.1f%%',startangle=90,labeldistance=1.04,pctdistance=0.9,frame=True)

ax.axis('equal')

plt.title("Poverty rates according to states",fontsize=15,color="B")

plt.show()