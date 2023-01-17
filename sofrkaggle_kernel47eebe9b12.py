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
#Вывод тренеровочной выборки 
tr=pd.read_csv('/kaggle/input/titanic/train.csv')
tr
#Вывод статистических данных
print(tr.describe())
#Вывод краткой сводки данных
tr.info()
#Замена значений "Sex" на булевые значения
for i in range(891):
    if tr.Sex[i]=='female':
        tr.Sex[i]=0
    elif tr.Sex[i]=='male':
        tr.Sex[i]=1
tr
#Вычисление количества мужчин на титанике
men=0
for i in range(891):
    men+=tr.Sex[i]
men=men/891
print(men)
#Поиск отсутствующих данных в выборке 
tr.isnull().sum()
#Поиск среднего возраста женщин 
trr=df.fillna(tr.mean())
res=0
age=0
for i in range(891):
    if trr.Sex[i]==0:
        res+=1
        age+=trr.Age[i]
print(res)
print(age)
srage=age/res
print(srage)
#Вычисление порта с наибольшим количеством пассажиров
S=0
Q=0
C=0
for i in range(891):
    if trr.Embarked[i]=='S':
        S+=1
    if trr.Embarked[i]=='C':
        C+=1
    if trr.Embarked[i]=='Q':
        Q+=1   
print(S,'\n',C,'\n',Q)
#Вычисление среднего возраста выживших женщин
res=0
age=0
for i in range(891):
    if trr.Survived[i]==1:
        if trr.Sex[i]==0:
            res+=1
            age+=trr.Age[i]
print(res)
print(age)
srv=age/res
print(srv)
#Вычисление распределения пассажиров по каютам
one=0
two=0
three=0
for i in range(891):
    if trr.Pclass[i]==1:
        one+=1
    if trr.Pclass[i]==2:
        two+=1
    if trr.Pclass[i]==3:
        three+=1   
print(one,'\n',two,'\n',three)
#Вычисление процента пассажиров 3 класса
one=0
two=0
three=0
for i in range(891):
    if trr.Pclass[i]==1:
        one+=1
    if trr.Pclass[i]==2:
        two+=1
    if trr.Pclass[i]==3:
        three+=1   
sum=one+two+three
print(sum)
prockl=three/sum
print(prockl)
#количество уникальных титулов в выборке "train"
a={}
for i in range(891):
    titul=list(trr.Name[i].split())
    for j in range(len(titul)):
        if '.' in titul[j]:
            if titul[j] in a:
                a[titul[j]]+=1
            else:
                a[titul[j]]=1
print(a)
#количество уникальных титулов в выборке "test"
trrr=pd.read_csv('/kaggle/input/titanic/test.csv')
b={}
for i in range(418):
    titul=list(trrr.Name[i].split())
    #print(titul)
    #print(len(titul))
    for j in range(len(titul)):
        if '.' in titul[j]:
            if titul[j] in b:
                b[titul[j]]+=1
            else:
                b[titul[j]]=1
print(b)