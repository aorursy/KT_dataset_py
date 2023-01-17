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
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.info()
data.head(10)
data.describe()
data.corr()
f,ax= plt.subplots(figsize=(13,13))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt=".1f",ax=ax)

plt.show()
data.target.value_counts()
sns.countplot(x="target", data=data, palette="bwr")

plt.show()
sagliklisayisi = len(data[data.target == 0])

hastasayisi = len(data[data.target == 1])

pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
pd.crosstab(data.sex,data.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)], c="red")

plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.title("age-heart rate corrolation")

plt.show()
data.plot(kind="line", )
# example of tuple 

def tuple_ex():

    t=(1,2,3)

    return t

x,y,z=tuple_ex()

print("a:",x,"\nb:",y,"\nc:",z)
#local and global scope

x= 5 #global scope

def random_method():

    y=x**2 #local scope

    return y

print(random_method())

print(x)    
#NESTED FUNCTIONS



def calculation(a,b):

    def nested_calculation():

        y=a+b

        return y

    return nested_calculation()+8

print(calculation(4,2))
# DEFAULT AND FLEXIBLE ARGUMENTS

""" default argument example"""

def mayo(a,b=4,c=3):

    y=a+b+c

    return y

print(mayo(2))
#flexible argument example



def rize(*args):

    for i in args:

        print(i*2)

rize(3)

rize(15)



def ordu(**kwargs):

    for key,value in kwargs.items():

        print(key,"",value)

ordu(Hakki="Turk", Rasim="Laz", Asim="Cerkez")



def ulkeler_ve_baskentleri(**kwargs):

    for key,value in kwargs.items():

        print(key,"",value)



ulkeler_ve_baskentleri(Rusya="Moskova", Ukrayna="Kiev", Belarus="Minsk", Estonya="Talinn")

    
#lambda function example



manhattan = lambda x: x + 3

print(manhattan(1))



kk= lambda y: y**2

print(kk(3))
#anonymous function example

yas_listesi=[20,21,22,23]

x=map(lambda a:a-10,yas_listesi)

print(list(x))



tbm_sayisi=[10,9,8,7]

y= map(lambda x:x-1,tbm_sayisi)

print(list(y))



son_ornek=[13,14,15]

c=map(lambda x:x**2,son_ornek)

print(list(c))



ins_son_ornek=[1,2,3]

a=map(lambda b:b+3,ins_son_ornek)

print(list(a))
#zip function example

list1=[20,21,22,23]

list2=[1,2,3,4]

z=zip(list1,list2)

z_list=list(z)

print(list(z))



list1=[20,21,22,]

list2=[1,2,3,4]

z=zip(list1,list2)

print(list(z))



data.info()