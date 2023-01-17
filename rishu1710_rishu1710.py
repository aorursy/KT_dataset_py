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

spinem = pd.read_csv("../input/spinem.csv")
import math as mt

K = [2,3,4,10,11,12,20,25,30]

m1 = 4

m2 = 12

c1 = [4]

c2 = [12]

l = len(K)

for i in range(l):

    if(abs(K[i]-m1) < abs(K[i]-m2)):

        if(K[i] == m1):

            continue

        c1.append(K[i])

    else:

        if(K[i] == m2):

            continue

        c2.append(K[i])   

l1 = len(c1)

l2 = len(c2)

m1 = 0

m2 = 0

for j in range(l1):

    m1=m1+c1[j]

m1=mt.ceil(m1/l1)    

for k in range(l2):

    m2=m2+c2[k]

m2=mt.ceil(m2/l2)

print(m1)

print(m2)

c1=[]

c2=[]

for i in range(l):

    if(abs(K[i]-m1) < abs(K[i]-m2)):

        c1.append(K[i])

    else:

        c2.append(K[i])  

l1 = len(c1)

l2 = len(c2)

m1 = 0

m2 = 0

for j in range(l1):

    m1=m1+c1[j]

m1=mt.ceil(m1/l1)    

for k in range(l2):

    m2=m2+c2[k]

m2=mt.ceil(m2/l2)

print(m1)

print(m2)

c1=[]

c2=[]

for i in range(l):

    if(abs(K[i]-m1) < abs(K[i]-m2)):

        c1.append(K[i])

    else:

        c2.append(K[i])          

print(*c1, sep = ", ")

print("---")

print(*c2, sep = ", ")