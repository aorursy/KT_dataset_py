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
df=pd.read_csv('/kaggle/input/titanic/train.csv')
df
print(df.describe())
df.info()
df
for i in range(891):
    if df.Sex[i]=='female':
        df.Sex[i]=0
    elif df.Sex[i]=='male':
        df.Sex[i]=1
df
res=0
for i in range(891):
    res+=df.Sex[i]
res=res/891
print(res)
df.isnull().sum()
dff=df.fillna(df.mean())
dff
dff.info()
res=0
age=0
for i in range(891):
    if dff.Sex[i]==0:
        res+=1
        age+=dff.Age[i]
print(res)
print(age)
vspom=age/res
print(vspom)
s=0
c=0
q=0
for i in range(891):
    if dff.Embarked[i]=='S':
        s+=1
    if dff.Embarked[i]=='C':
        c+=1
    if dff.Embarked[i]=='Q':
        q+=1   
print(s,c,q,sep="   ")
res=0
age=0
for i in range(891):
    if dff.Survived[i]==1:
        if dff.Sex[i]==0:
            res+=1
            age+=dff.Age[i]
print(res)
print(age)
vspom=age/res
print(vspom)
uno=0
dos=0
trees=0
for i in range(891):
    if dff.Pclass[i]==1:
        uno+=1
    if dff.Pclass[i]==2:
        dos+=1
    if dff.Pclass[i]==3:
        trees+=1   
print(uno,dos,trees,sep="   ")
uno=0
dos=0
trees=0
for i in range(891):
    if dff.Pclass[i]==1:
        uno+=1
    if dff.Pclass[i]==2:
        dos+=1
    if dff.Pclass[i]==3:
        trees+=1   
sum=uno+dos+trees
print(sum)
otvet=trees/sum
print(otvet)
d={}
m={}
for i in range(891):
    titul=list(dff.Name[i].split())
    #print(titul)
    #print(len(titul))
    for j in range(len(titul)):
        if '.' in titul[j]:
            if titul[j] in d:
                d[titul[j]]+=1
            else:
                d[titul[j]]=1
print(d)
print('L. - попала в словарь случайно. Она не титул')
print('17')
dfff=pd.read_csv('/kaggle/input/titanic/test.csv')
dfff
d={}
m={}
for i in range(418):
    titul=list(dfff.Name[i].split())
    #print(titul)
    #print(len(titul))
    for j in range(len(titul)):
        if '.' in titul[j]:
            if titul[j] in d:
                d[titul[j]]+=1
            else:
                d[titul[j]]=1
print(d)