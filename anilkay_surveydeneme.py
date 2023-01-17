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
data=pd.read_csv("/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv")

data.head()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
questions=data.iloc[0:1,:]

for column in data.columns:

    print(questions[column].values[0],"  ",column)
answers=data.iloc[1:,:]
plt.figure(figsize=(10,10))

ax=sns.countplot(data=answers,x="Q1")

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()

plt.figure(figsize=(14,10))

ax=sns.countplot(data=answers,x="Q23")

#ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

#plt.tight_layout()

set(answers["Q26"])
len(answers[answers["Q26"]=="Definitely yes"])
len(answers[answers["Q26"]=="Definitely yes"])/len(answers)
print("Number of: ",answers["Q36_Part_9"].dropna().count())

print("Percantage: ",answers["Q36_Part_9"].dropna().count()/len(answers))
print("Number of: ",answers["Q38_Part_19"].dropna().count())

print("Percantage: ",answers["Q38_Part_19"].dropna().count()/len(answers))
print("Number of: ",answers["Q38_Part_18"].dropna().count())

print("Percantage: ",answers["Q38_Part_18"].dropna().count()/len(answers))
print("Number of: ",answers["Q38_Part_11"].dropna().count())

print("Percantage: ",answers["Q38_Part_11"].dropna().count()/len(answers))
set(answers["Q48"])
withoutnaQ48=answers["Q48"].dropna()

withoutnaQ48.value_counts()
prog_list=[]

for i in range(1,19):

    for  values in answers["Q16_Part_"+str(i)].dropna().values:

         prog_list.append(values)

plt.figure(figsize=(14,10))            

ax=sns.countplot(x=prog_list)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()
plt.figure(figsize=(14,10)) 

ax=sns.countplot(x=answers["Q18"])

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()
print(answers["Q49_Part_7"].dropna().count())

all_Q49=0

for i in range(1,11):

    all_Q49+=answers["Q49_Part_"+str(i)].dropna().count()

print("Percantage: ",answers["Q49_Part_7"].dropna().count()/all_Q49*100)    
fortunes=[] #Quentin Fortune was a Manchester United Player. 

for i in range(1,19):

    for value in answers["Q19_Part_"+str(i)].values:

        fortunes.append(value)

        

plt.figure(figsize=(14,10)) 

ax=sns.countplot(x=fortunes)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()        