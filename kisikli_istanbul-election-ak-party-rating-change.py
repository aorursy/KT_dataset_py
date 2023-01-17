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



# Any results you write to the current directory are saved as outpu
dtf = pd.read_csv("../input/istanbul_secim_akp3.csv")

dictionary = {"acilan_sandik":[],"oy_orani":[] } 

dtf1 = pd.DataFrame(dictionary) #to edit the dataset, create a new dataframe



liste = dtf.values.tolist()

i=0

while(i!=26): 

    j=0

    a=""

    while(liste[i][0][j]!=";"):

        a=a+liste[i][0][j]

        j+=1

    dtf1.loc[i,"acilan_sandik"]=a

    b=liste[i][0][j+1:]

    dtf1.loc[i,"oy_orani"]=b

    i+=1



dtf1=dtf1[::-1]

dtf1 = dtf1.astype("float64")

print(dtf1)
print(dtf1.corr())

sns.heatmap(dtf1.corr(),annot=True,linewidth=.5,fmt=".1f")

plt.show()
plt.plot(dtf1.acilan_sandik,dtf1.oy_orani,color="red")

plt.xlabel("Rate of opened ballot boxes")

plt.ylabel("rating of AK Party")

plt.title("31 March AKP-İstanbul Rating Change")

plt.show()
plt.scatter(dtf1.acilan_sandik,dtf1.oy_orani,color="blue",alpha = 0.5)

plt.xlabel("Rate of opened ballot boxes")

plt.ylabel("rating of AK Party")

plt.title("31 March AKP-İstanbul Rating Change")

plt.show()