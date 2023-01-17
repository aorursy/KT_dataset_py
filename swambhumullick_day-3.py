# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

Iris = pd.read_csv("../input/iris/Iris.csv")

print(Iris)
Iris = pd.read_csv("../input/iris/Iris.csv")

Iris.head()
Iris.columns
Iris["Species"].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt

sns.FacetGrid(Iris, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend();

plt.show();
x=6

y=3

z=5.5

w=2.1

d1=Iris['SepalLengthCm']-x

d2=Iris['SepalWidthCm']-y

d3=Iris['PetalLengthCm']-z

d4=Iris['PetalWidthCm']-w

dist=d1*d1+d2*d2+d3*d3+d4*d4

print(dist)
arr={}

x=6

y=5

d1=Iris['SepalLengthCm']

d2=Iris['SepalWidthCm']

l=len(Iris)

c=0

for i in range(0,l):

    arr[i]=((d1-x)**2+(d2-y)**2)**0.5

min=0



for i in range(0,l):

    if(arr[i]<arr[min]):

        min=i

        

s10=Species[min]

       