# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/heart.csv")

data.describe()
plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True)

data.head(50)
data.oldpeak.plot(kind="line",color="b",label="oldpeak",linewidth=2,grid=True,linestyle = "-" , alpha = 0.5,figsize=(20,20))

data.slope.plot(kind="line",color="g",label="slope",linewidth=2,grid=True,linestyle = "-" , alpha = 0.5)

plt.xlabel("data size",fontsize="xx-large")

plt.ylabel("frequency",fontsize="xx-large")

plt.legend(loc="best",fontsize="xx-large")

data.plot(kind="scatter", x = "oldpeak" , y = "slope",alpha=0.8 , color ="g",figsize=(10,10))
data.oldpeak.plot(kind="hist",bins=50,figsize=(15,15))
boolean1 = data["oldpeak"] > 1

boolean2 = data["oldpeak"] < 4

boolean3 = data["age"] > 75

age =  int(data[boolean1 & boolean2 & boolean3].age) #then, can we pick up the index of the element? in this output its 144 ,  can I get this value from somewhere and then can I use it to access the element like data[144] ?



data[data["age"] == age]
for element in data[["age"]].iterrows():

    print(element)
for index,element in data[data["age"] > 70].iterrows():

    print(index,"---------\n",element)
data["age"][data["age"] > 70]



for index,element in data["age"][data["age"] > 70].items():

    print(index,"  ",element)
data["age"][data["age"] > 70][25] #access the elements index 
data[data["age"] > 70]

print(type(data["age"][data["age"] > 70]) , "\n" , type(data[data["age"] > 70]) )
data["age"][25] #series has index access but dataframe object has not