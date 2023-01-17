import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('../input/z4-stat/Z4.txt', sep=" ", header=None)

df.columns = ['x1', 'x2','x3', 'x4', 'y1','y2', 'd'] 

df=df.drop('d',axis=1)

df
n=52

q=4

# Без x2:

# df=df.drop('x2',axis=1)

# q=3 

p=2
X1zm=np.array(df['x1']).sum()/n

X1zm
sigmaX1z=np.sqrt(np.array((df['x1']-X1zm)**2).sum()/(n-1))

sigmaX1z
X1z=(df['x1']-X1zm)/sigmaX1z
X2zm=np.array(df['x2']).sum()/n

X2zm



sigmaX2z=np.sqrt(np.array((df['x2']-X2zm)**2).sum()/(n-1))

sigmaX2z



X2z=(df['x2']-X2zm)/sigmaX2z
X3zm=np.array(df['x3']).sum()/n

X3zm



sigmaX3z=np.sqrt(np.array((df['x3']-X3zm)**2).sum()/(n-1))

sigmaX3z



X3z=(df['x3']-X3zm)/sigmaX3z
X4zm=np.array(df['x4']).sum()/n

X4zm



sigmaX4z=np.sqrt(np.array((df['x4']-X4zm)**2).sum()/(n-1))

sigmaX4z



X4z=(df['x4']-X4zm)/sigmaX4z
Y1zm=np.array(df['y1']).sum()/n

Y1zm



sigmaY1z=np.sqrt(np.array((df['y1']-Y1zm)**2).sum()/(n-1))

sigmaY1z



Y1z=(df['y1']-Y1zm)/sigmaY1z
Y2zm=np.array(df['y2']).sum()/n

Y2zm



sigmaY2z=np.sqrt(np.array((df['y2']-Y2zm)**2).sum()/(n-1))

sigmaY2z



Y2z=(df['y2']-Y2zm)/sigmaY2z

df['x1']=X1z

# df['x2']=X2z

df['x3']=X3z

df['x4']=X4z

df['y1']=Y1z

df['y2']=Y2z

df
R=np.array(df.corr(method='pearson'))

df.corr(method='pearson')
R11=R[0:q,0:q]

R11
R12=R[0:q,q:q+p]

R12
R21=R[q:q+p,0:q]

R21
R22=R[q:q+p,q:q+p]

R22
from numpy.linalg import inv

R22_1=inv(R22)

R22_1
R11_1=inv(R11)

R11_1
C=np.dot(np.dot(np.dot(R22_1,R21),R11_1),R12)

C
from numpy.linalg import eig

Cc=eig(C)

Cc
L12=Cc[0][0]

L12
L22=Cc[0][1]

L22
r1=np.sqrt(L12)

r1
r2=np.sqrt(L22)

r2
B1=Cc[1][:,0]

B1
B2=Cc[1][:,1]

B2
B1T=np.array([B1]).transpose()

B1T
B2T=np.array([B2]).transpose()

B2T
print(np.dot(C,B1T))

np.dot(L12,B1T)

# print(np.dot(C,B2T))

# np.dot(L22,B2T)
A1=np.dot(np.dot(R11_1,R12),B1T)

A1
A2=np.dot(np.dot(R11_1,R12),B2T)

A2