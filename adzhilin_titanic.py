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
sur = 0

i = 0
for i in range(891):
    if df.Survived[i]== 0:
        sur +=1
print(sur)
proc = sur * 100 / 891
round(proc, 0)


i = 0
devka = 0
men = 0
for i in range(891):
    if df.Sex[i] == 'female':
        devka +=1
    elif df.Sex[i] == "male":
        men +=1
print(devka, men, end= " ")

df
dff = df.fillna(df.mean())
    
sum = 0
for i in range(891):
    if dff.Sex[i] == "female":
        sum += dff.Age[i]
print(sum)
sred = sum / 314
round(sred, 1)  
    
s = 0
c = 0
q = 0
for i in range(891):
    if dff.Embarked[i] == "S":
        s += 1
    elif dff.Embarked[i] == "C":
        c += 1
    else:
        q += 1
print(s, c, q, end = " ")
        

dff
sur = 0
summ = 0
for i in range(891):
    if dff.Survived[i] == 1:
        if dff.Sex[i] == "female":
            sur += 1
            summ += dff.Age[i]
print(sur, summ, end = " ")
proc = round(summ/sur, 2)
print(proc)
dff
per = 0
vtor = 0
tret = 0
for i in range(891):
    if dff.Pclass[i] == 1:
        per += 1
    elif dff.Pclass[i] == 2:
        vtor += 1
    else:
        tret += 1
print(per, vtor, tret, end = " ")


round(tret*100/891, 0)
dff
df.isnull().sum()
din ={}
for i in range(891):
    titul=list(dff.Name[i].split())
    for j in range(len(titul)):
        if '.' in titul[j]:
            if titul[j] in din:
                din[titul[j]]+=1
            else:
                din[titul[j]]=1
print(din)