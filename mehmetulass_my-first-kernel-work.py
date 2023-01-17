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
data = pd.read_csv('../input/heart.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.15, fmt= '.2f',ax=ax)

plt.show()
data.head(15)
data.tail(14)
data.columns
data.chol.plot(kind="line",color="r",label="chol",linewidth=1,alpha=1.0,grid=True,linestyle=":")

data.fbs.plot( color="g",label="fbs",linewidth=1,alpha=1.0,grid=True,linestyle="-.")

plt.legend("upper right") 

plt.title("line plot")

plt.xlabel("chol")

plt.ylabel("fbs")

plt.show()
data.plot(kind="scatter",x="oldpeak",y="thalach",alpha=0.5,color="red",grid=True)

plt.xlabel("oldpead")

plt.ylabel("thalach")

plt.title("scatter plot")

plt.show()
data.age.plot(kind="hist",bins=123,figsize=(15,15))

plt.show()
data.age.plot(kind = 'hist',bins = 123)

plt.clf()
dictionary={"İstanbul":"Turkey","Berlin":"Germany"}

print(dictionary.keys())

print(dictionary.values())
dictionary["İstanbul"]="Ankara"

print(dictionary)

dictionary["New York"]="USA"

print(dictionary)
del dictionary["Berlin"]

print(dictionary)
print('Canada' in dictionary)   

print('İstanbul' in dictionary)   
dictionary.clear()

print(dictionary)

#del dictionary

# dictionary
age=data["age"]

print(age)

print(type(age))
age_dataframe=data[["age"]]

print(age_dataframe)

print(type(age_dataframe))
X=data["trestbps"]<150

print(data[X])
data[np.logical_and(data['age']>55,data['sex']>0)]
data[(data["oldpeak"]<2) & (data["trestbps"]<142)]
i=0

while i!=5:

    print("i is",i)

    i+=1

print("i is equal:",i)
list=[1,3,5,7,9,11,14,53]

for i in list:

    print("i is:",i)

  

print("-------------")





for index,number in enumerate(list):

    print(index,"is",number)
dictionary={"Teacher":"School","Soldier":"Military"}

for key,value in dictionary.items():

    print(key,":",value)

    

    

print("---------------")

for index,value in data[['trestbps']][0:5].iterrows():

    print(index," : ",value)

    

    