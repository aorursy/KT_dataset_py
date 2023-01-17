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
data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

data.head()
data.corr()
f,ax = plt.subplots(figsize=(5,5))

sns.heatmap(data.corr(), annot=True, linewidths=.4, fmt=".1f", ax=ax)
data.head(1)
#line plot

data.Confirmed.plot(kind="line", color="pink", label="Confirmed")

plt.legend(loc="upper right")

plt.show()
#scatter plot



data.plot(kind="scatter", x="Confirmed", y="Deaths", color="black", alpha=0.2)

plt.xlabel="confirmed"

plt.ylabel="Deaths"

plt.title="Deaths-Confirmed Corona"
#histogram plot

data.Recovered.plot(kind="hist",bins=20, color="orange")

plt.show()
# user defined func

def tuble_exa():

    t = ("m","e","r","v","e")

    return t

m,e,r,v,e=tuble_exa()

print(m,e,r,v,e)
#lambda func



multi = lambda m : m**2+m*3

multi(3)
#lambda func



calculator = lambda m,r,v: m**5-r**0*v**3

calculator(3,6,4)
#anonymous func



listexa={1,3,5,7}

y = map(lambda x: x**2+5, listexa)

print(list(y))
#iterators



course_name="DataScience"

first=iter(course_name)

print(next(first),next(first),next(first),next(first))

print(*first)
#zip metodu



index=[1,2,3,4,5]

data=["m","e","r","v","e"]

zipp = zip(index,data)

z = list(zipp)

print(list(z))
#unzip

unzipp = zip(*z)

un1,un2= list(unzipp)

print(un1,un2)
#list comprehension



lst = [3,4,5]

list_2 = [i*3 for i in lst]

print(list_2)
lstn = [2,4,8,78]

lstnev = [i+5 if i==3 else i*2 if i>5 else i-3 for i in lstn]

print(lstnev)
data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

data.head()
sumdata = sum(data.Confirmed)/len(data.Confirmed)

print("Avg of Confirmed",sumdata)

data["Teshis"]=["high" if i>sumdata else "low" for i in data.Confirmed]



data.loc[210:220,["Confirmed", "Teshis"]]