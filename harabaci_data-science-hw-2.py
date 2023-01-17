# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read the Iris data from Csv

iris=pd.read_csv("/kaggle/input/iris/Iris.csv")
# Display content of Iris data 

iris.info()
#Data overview

iris.describe()
#Data correlation

iris.corr()
import matplotlib.pyplot as plt
iris
iris.columns
iris["Species"].unique()
seri=iris["SepalLengthCm"]

df=iris[["SepalLengthCm"]]
print(type(seri))

print(type(df))
iris[iris.SepalLengthCm>7.4]
iris[(iris.SepalLengthCm>7.4) & (iris.PetalLengthCm<6.7)]
iris[np.logical_and(iris["SepalLengthCm"]>7.4, iris["PetalLengthCm"]<6.7)]
#Add new features to the DataFrame

iris["TotalLenght"]=iris["SepalLengthCm"]+iris["PetalLengthCm"]

iris["TotalWidth"]=iris["SepalWidthCm"]+iris["PetalWidthCm"]

iris
#Adding new column to Dataframe with using if and for loop(list comprehension)

iris["Compare"]=[1 if i>14 or j>=6 else 0 for i,j in zip(iris.TotalLenght,iris.TotalWidth)]

iris
iris[iris.Compare==1]
list1=[i for i in iris.SepalLengthCm]

list2=[i for i in iris.SepalWidthCm]

zip_Sepal=zip(list1,list2)

print(zip_Sepal)
list_Sepal=list(zip_Sepal)

print(list_Sepal)
print(list_Sepal[0])

i,j=list_Sepal[0]

print("SepalL:",str(i),"SepalW:",str(j))
# iteration example

it=iter(list_Sepal)

print(next(it))

print(next(it))

print(*it)
un_zip=zip(*list_Sepal)

u_list1,u_list2=list(un_zip)

print(list(u_list1[0:2]), u_list2[0:2])
#Anonymous function

u_list1=list(u_list1)

u_list1

u_list1_double=map(lambda i:i*2, u_list1)

list(u_list1_double)