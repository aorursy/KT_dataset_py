# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import seaborn as sns
from scipy.stats import norm
log1=[]
log2=[]
log3=[]
with open('/kaggle/input/log1.txt') as file:
    log1=[x.split(' ') for x in file.readlines()]
with open('/kaggle/input/log2.txt') as file:
    log2=[x.split(' ') for x in file.readlines()]
with open('/kaggle/input/log3.txt') as file:
    log3=[x.split(' ') for x in file.readlines()]

cnt=0
Y=[]
last=0
avg=0
print('Log sizes',len(log2),len(log1),len(log3))
for i in range(len(log2)):
    if log2[i][0]!=log1[i][0] or log3[i][0]!=log1[i][0]:
        #print(' '.join(log1[i-1]),' '.join(log2[i-1]))
        #print(' '.join(log1[i]),' '.join(log2[i]))
        #print(' '.join(log1[i+1]),' '.join(log2[i+1]))
        cnt+=1
        Y.append(i-last)
        last=i
        avg+=i-last

sns.scatterplot(range(len(Y)),Y)
mean,std=norm.fit(data=np.array(Y))
print(f'Total mismatches {cnt}')
plt.ylim(0,800)
print(f'mean, std: {mean},{std}')
from random import choices
q_table=[[0]*2]*2
s=0
for i in range(200):
    ep=choices([0,1])
    a=0
    if ep==2:
        if q_table[s][0]<q_table[s][1]:
            a=1
    else:
        a= choices([0,1])[0]
    snew=s
    r=1
    if a==1:
        snew=(s+1)%2
        r=0
    q_table[s][a]=0.5*q_table[s][a]+0.5*(r+0.9*max(q_table[snew][0],q_table[snew][1]))
    s=snew
print(q_table)