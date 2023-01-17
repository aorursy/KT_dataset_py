# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np
#creation/ initialization



#list



a = np.array([1,2,3,4,5,6])



#all zeros



b = np.zeros(10)



#all 5's matrix of size 4X4



c = np.full((4,4),5)
a

b
c
#creating sequence



a =  np.arange(10,100,5)



#equal spaced 10 numbers between 1 & 2



b = np.linspace(start=1, stop=2,num=10)



#random whole numbers between 5 and 10



c = np.random.randint(5,10,size=(3,4))



#random 10 numbers between 0 and 1



d = np.random.rand(10)
d
#access numpy array



h = np.random.randint(5,10,size=(5,5))

h
#2nd row onwards rest of all

h[2:]
h[1:4]
h[:,1:3]
#shape & reshape



i = np.random.randint(1,5,size=(4,5))
i
i.shape
i.size
i.ndim
j = i.reshape(2,10)
j
j.shape
k = j.ravel()
k
k.shape
m =  np.arange(5)
m
np.sum(m)
#pandas basics
import pandas as pd

s1 = pd.Series(data=[1,2,3,4], index=["a","b","c","d"])
s1
s1["a"]
s2 = pd.Series(data=[11,12,13,14],index=["a","b","c","d"])
s2

df1 = pd.DataFrame({'col1':s1,'col2':s2})
df1