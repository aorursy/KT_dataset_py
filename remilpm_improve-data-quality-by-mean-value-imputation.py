import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

Tit1=pd.read_csv("../input/Titanic.csv")

# Check the sum of null values

Tit1.isnull().sum()

#Sample of Age  null values

Tit1[Tit1['Age'].isnull()].head()
#Check the count of valid age values

Tit2=Tit1.copy()

#Get the count

Tit2['Age'].count()

#Apply Mean imputataion 

#Find the mean age and replace all the null values with mean value

Mean_Age=round(Tit2['Age'].sum()/Tit2['Age'].count(),2)

Mean_Age
#Filter out valid age

Cond1= Tit2['Age']>0

Tit3=Tit2[Cond1].copy()

Tit3.shape
#Remove the valid values to facilitate age mean value assignment

#Find the invalid age

Tit4=pd.concat([Tit3,Tit2]).drop_duplicates(keep=False)

Tit4.head()
#Invalid age values, total values

Tit4.shape, Tit2.shape
#Now substitute with mean value

Tit4['Age']=Mean_Age

Tit4.shape
Tit4.head()
#Now check for Age assignment

cond2=Tit4['Age']==29.7

Tit4=Tit4[cond2]

Tit4.shape
# Sum up both the dataset

Tit5=pd.concat([Tit3,Tit4]).drop_duplicates(keep=False)

Tit5.shape
#Successfully replaced Age with mean values

Tit5[Tit5['Age'] >0].count()
Tit5.isnull().sum()