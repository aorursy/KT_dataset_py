

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from scipy import stats

from mlxtend.preprocessing import minmax_scaling

import seaborn as sns

import matplotlib.pyplot as plt

print(os.listdir("../input"))

Norm1=pd.read_csv("../input/Titanic.csv")

Norm1.tail()





cond1=(Norm1['Age']>0)

cond1=(Norm1['Fare']>0)

Norm2= Norm1[cond1 & cond1]

Norm2.head()



#Remove the string columns

Norm3=Norm2[['PassengerId','Survived','Pclass','Age','SibSp','Parch','Fare']]

Norm3.head()
#Normalizing the dataframe

Norm4=((Norm3-Norm3.min())/(Norm3.max()-Norm3.min()))*20

Norm4.head()