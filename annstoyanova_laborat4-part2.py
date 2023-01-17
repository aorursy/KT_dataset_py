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

x = np.random.normal(0, 2.5, (20, 4)) 

x

x1=x[:5]

x1
x2=x[:,-2:]

x2
x3=x[4,:]

x3



x_mean=x3.mean()

x_mean
x1_mean=x.mean(axis=1)

x1_mean#по всем строкам

a=x[:,0]

maxx=np.max(a)

maxx
e=x[0,:]

e1=np.sum(e) #summa

e2=e1*2 #udvoennaya summa

r=x[:,-1]

r1=np.sum(r)

if e2>r1:

    print('Verno')

else:

    print('Neverno')

maxx1=np.max(x)

maxx1

b=print(maxx1-x[:,:])

b#novaya matrica

maxx1=np.max(x)

maxx1

b=maxx1-x[:,:]

r=b[(b[:, 0]>0) & (b[:,-1]>0),:]

r



    
maxx1=np.max(x)

maxx1

b=maxx1-x[:,:]

t=print(b[:, np.mean(b, axis=0)>0.5]) 

t
y= b.transpose()

y

Y1 = np.delete(y,[0,1,2,3,4] , axis=1)

Y1
b = np.ones(4)



C2 = np.append(b[:, np.newaxis], Y1, axis=1) # альтернативный способ

print(C2)
Y1.shape
Z = np.zeros((4, 18))

E = np.vstack([Y1, Z])

E