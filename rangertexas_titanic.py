import pandas as pd

import numpy as np

import sklearn as ml

import matplotlib as plt



df=pd.read_csv("../input/train.csv")

print(df.columns)
df.head()
ModeAge=df.Age[df.Age.isnull()==False].mode()

plt.pyplot.hist(df.Age[df.Age.isnull()==False])  
label=df.Survived

n_survived=df[df.Survived==1].Survived.count()

total=df.Survived.count()

perc_survived=n_survived/total

print ('percentage of survived people =',perc_survived)
plt.pyplot.plot(df.Survived,df.Fare,'ro')