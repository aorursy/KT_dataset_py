# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/titanic/train_and_test2.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
df=pd.read_csv("../input/covid19-in-india/covid_19_india.csv")

df.head()
l=[]
for i in df["State/UnionTerritory"]:
    if i not in l:
        l.append(i)
print(len(l))        
j=[x for x in range(0,40)]
j1=list(zip(l,j))
print("Select Code of Your City")
print(j1)

g=int(input("please enter your state code")) 

kk=[]
for i in df["State/UnionTerritory"]:
    if i not in l:
        kk.append(i)
for j in range(3):
    df1=df[df["State/UnionTerritory"] ==l[g]]

 
df1.head()
l=[]
l2=[]
for i in df1["Confirmed"]:
    l.append(i)
for k in range(len(l)-1):
    c=l[k+1]-l[k]
    l2.append(c)
sum(l2)   
x=l2
y=[x for x in range(len(l2))]
len(x)==len(y)
v=sum(l2)
print("Total Cases",v)
print("Here is Treding Graph")
import matplotlib.pyplot as plt
plt.plot(y,x)
print("Here X axis represent number of days")
print("Here Y axis represent number of cases")
plt.scatter(y,x)
