import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('../input/z5stat/Z5.txt', sep=" ", header=None)

df.columns = ['x','y','d'] 

df=df.drop('d',axis=1)

df
n=40
df['x'].sort_values()
A= set()

for j in range(40):

    for i in range(j+1,40):

        if (df['x'].iloc[j]==df['x'].iloc[i]) and not(df['x'].iloc[i] in A):

            A.add(df['x'].iloc[j])

A
df['y'].sort_values()

A= set()

for j in range(40):

    for i in range(j+1,40):

        if (df['y'].iloc[j]==df['y'].iloc[i]) and not(df['y'].iloc[i] in A):

            A.add(df['y'].iloc[j])

A
df['x1']=df['x']

df['y1']=df['y']

df['x']=df['x'].rank()

df['y']=df['y'].rank()

df
d2=np.array((df['x']-df['y'])*(df['x']-df['y'])).sum()

d2
Ty=10

rc=1-((n**3-n)/6-d2-Ty)/np.sqrt((n**3-n)*((n**3-n)/6-2*Ty)/6)

rc
np.sqrt((n-2)/(1-rc*rc))
t_st=np.abs(rc)*np.sqrt((n-2)/(1-rc*rc))

t_st
df=df.sort_values('x')

df
P_arr=np.zeros(n)

for i in range(n):

    for j in range(i+1,n):

        if df['y'].iloc[j]>df['y'].iloc[i]:

            P_arr[i]+=1

P_arr
P=P_arr.sum()

P
Q_arr=np.zeros(n)

for i in range(n):

    for j in range(i+1,n):

        if df['y'].iloc[j]<df['y'].iloc[i]:

            Q_arr[i]+=1

Q_arr
Q=Q_arr.sum()

Q
S=P-Q

S
Vy=17

rk=S/(np.sqrt(n*(n-1)*(n*(n-1)/2-Vy)/2))

rk
1.959964*np.sqrt(2*(2*n+5)/(9*n*(n-1)))