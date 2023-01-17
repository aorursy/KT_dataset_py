# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

df.head()
df.info()
df.corr()
df.trestbps.plot(kind = "line",color = "r",label = "trestbps",alpha = 0.5,grid = True,linewidth = 1,figsize = (12,12))

plt.xlabel("person")

plt.ylabel("trestbps")

plt.title("person-trestbps")

plt.legend()

plt.show()
df.plot(kind = "scatter",x = "thalach",y = "chol",color = "blue",alpha = 0.5)

plt.xlabel("thalach")

plt.ylabel("chol")

plt.title("thalach-chol corr")

plt.show()
df.cp.plot(kind = "hist",bins = 8,figsize = (7,7))

plt.show()
dic = {63 : 145, 37 : 130,41:130}

print(dic.keys())

print(dic.values())
dic[56] = 120

print(dic)

del dic[41]

print(dic)

print(41 in dic)

dic.clear()

print(dic)
a = np.logical_and(df.age>65,df.trestbps<120)

b = df[a]

b
liste = list(df.age[np.logical_and(df.age>65,df.trestbps<120)])

print(liste)

i = 0

while liste[i] > 66:

    for index,value in b[["sex"]][i :i+1].iterrows():

        print(value["sex"]) 

    i +=1

    

b