import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

mr=pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv",encoding='ISO-8859-1')

mr

ot=pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv",encoding='ISO-8859-1')

ot

qo=pd.read_csv("../input/kaggle-survey-2019/questions_only.csv",encoding='ISO-8859-1')

qo

ss=pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv",encoding='ISO-8859-1')

ot

x=mr['Q1'].value_counts()

x.head()

fig,ax=plt.subplots(figsize=(20,10))

plt.bar(x.index,x.values,width=.5)

plt.title("Age",size=20)

plt.show()

x1=mr['Q2'].value_counts()

x1

fig,ax=plt.subplots(figsize=(20,10))

plt.bar(x1.index,x1.values,width=.5)

plt.show()



slices = [7,2]

activities = ['male','female']

cols = ['c','m']

outside = (0, 0.1) 

plt.pie(slices,labels=activities,colors=cols,startangle=90,explode=outside,shadow=True)

plt.legend()

plt.show()







x2=mr['Q8'].value_counts()

x2

fig,ax=plt.subplots(figsize=(30,15))

ax.set_xticklabels(x2.index, size=15,rotation=60)

plt.bar(x2.index,x2.values,width=.5)

plt.show()







x3=mr['Q13_Part_12'].value_counts()

x3

fig,ax=plt.subplots(figsize=(20,10))



plt.bar(x3.index,x3.values,width=.5)

plt.show()
