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
Iris.describe()
Iris.drop(columns='Species')
x=5.5

y=6.0

z=x-Iris['SepalLengthCm']

a=y-Iris['SepalLengthCm']

b=x-Iris['SepalWidthCm']

r=y-Iris['SepalWidthCm']

c=x-Iris['PetalLengthCm']

d=y-Iris['PetalLengthCm']

e=x-Iris['PetalWidthCm']

f=y-Iris['PetalWidthCm']

df_col1 = pd.concat([z,b,c,e],axis=1)

df_col2 = pd.concat([a,r,d,f],axis=1)

print(df_col1,df_col2)
import numpy as np

k=[2,3,4,10,11,12,20,25,30]

m1=10

m2=12

while(True):

    k1=[]

    k2=[]

    for i in range(0,len(k)):

        if(abs(k[i]-m1)<abs(k[i]-m2)):

            k1.append(k[i])

        else:

            k2.append(k[i])

    new_m1=np.mean(k1)

    new_m2=np.mean(k2)

   

    if(m1==new_m1 and m2==new_m2):

        break

    else:

        m1=new_m1

        m2=new_m2

print("First cluster:",k1)

print("Second cluster:",k2)