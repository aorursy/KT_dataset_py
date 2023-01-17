# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
Data=pd.read_csv("../input/StudentsPerformance.csv")

# Any results you write to the current directory are saved as output.
Data.head()
#Lets add columns of total score and average
Data["Total_Score"]=Data["math score"]+Data["reading score"]+Data["writing score"]
Data["Percentage"]=(Data["Total_Score"]/300)*100
#Some EDA
# Let's talk about gender ,Race,Parental education and lunch
Gender=Data["gender"]

#Race
Race=Data["race/ethnicity"]

#Some Parental Education
PE=Data["parental level of education"]

#Some Lunch Data
Lunch=Data["lunch"]
#Bar Charts of some categoricl variables

fig = plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((4,3), (0,0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((4,3), (0,1), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((4,3), (0,2), rowspan=1, colspan=1)
ax4 = plt.subplot2grid((4,3), (1,0), rowspan=1, colspan=2)
ax5 = plt.subplot2grid((4,3), (1,2), rowspan=1, colspan=1)

#sns.barplot(x=Gender.unique(), y=Gender.value_counts(), palette="rocket", ax=axs[0,0])
ax1.bar(Gender.unique(),Gender.value_counts())
ax1.set_xlabel('Gender Bar Chart')
ax2.bar(Race.unique(),Race.value_counts())
ax2.set_xlabel('Race Bar Chart')
ax3.bar(Lunch.unique(),Lunch.value_counts())
ax3.set_xlabel('Lunch Bar Chart')
ax4.bar(PE.unique(),PE.value_counts())
ax4.set_xlabel('Parental education Bar Chart')
ax5.bar(Data["test preparation course"].unique(),Lunch.value_counts())
ax5.set_xlabel('test preparation course')
plt.show()
#Some Result
print(Data[Data["Percentage"]==max(Data["Percentage"])])
#So 2 Female out of 3 got 100% marks.
print(Data[Data["Percentage"]==min(Data["Percentage"])])
#One Female got lowest percentage
#Plotting all the Distrbutions of all marks
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

axs[0,0].hist(Data["math score"],edgecolor="black")
axs[0,0].set_xlabel('Math Score Distribution')
axs[0,1].hist(Data["reading score"],edgecolor="black")
axs[0,1].set_xlabel('Reading Score Distribution')
axs[1,0].hist(Data["writing score"],edgecolor="black")
axs[1,0].set_xlabel('Writing Score Distribution')
axs[1,1].hist(Data["Percentage"],edgecolor="black")
axs[1,1].set_xlabel('Percentage Score Distribution')
plt.show()

#Distributions of score based on different categories
fig = plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((4,3), (0,0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((4,3), (0,1), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((4,3), (0,2), rowspan=1, colspan=1)
ax4 = plt.subplot2grid((4,3), (1,0), rowspan=1, colspan=2)
ax5 = plt.subplot2grid((4,3), (1,2), rowspan=1, colspan=1)


sns.boxplot(x=Data["gender"], y=Data["Total_Score"], palette=["m", "g"],ax=ax1)
sns.boxplot(x=Data["race/ethnicity"], y=Data["Total_Score"], palette=["m", "g"],ax=ax2)

sns.boxplot(x=Data["lunch"], y=Data["Total_Score"], palette=["m", "g"],ax=ax3)
sns.boxplot(x=Data["parental level of education"], y=Data["Total_Score"], palette=["m", "g"],ax=ax4)
sns.boxplot(x=Data["test preparation course"], y=Data["Total_Score"], palette=["m", "g"],ax=ax5)
plt.show()

MData=Data
#Labelling all categorical variables,Just some preprocessing before applying ML algorithms
MData["gender"]=MData["gender"].replace({"male":0,"female":1})
MData["test preparation course"]=MData["test preparation course"].replace({"none":0,"completed":1})
MData["lunch"]=MData["lunch"].replace({"standard":1,"free/reduced":0})
MData["race/ethnicity"]=MData["race/ethnicity"].replace({"group A":1,"group B":2,"group C":3,"group D":4,"group E":5})
MData["parental level of education"]=MData["parental level of education"].replace({"bachelor's degree":1,"some college":2,"master's degree":3,"associate's degree":4,"high school":5,"some high school":6})
#COORELATION
Corr=MData.corr()
sns.heatmap(Corr)
plt.show()
MData["Result"]=[1 if i>=40  else 0 for i in MData["Percentage"] ]
#Let's some machine learning
from sklearn.model_selection import train_test_split
X_Data=MData.loc[:,"gender":"test preparation course"]
Y_Data=MData["Result"]
X_train,X_test,Y_train,Y_test=train_test_split(X_Data,Y_Data,test_size=0.2,random_state=28)

#My Favourite
from sklearn.tree import DecisionTreeClassifier
DTree=DecisionTreeClassifier()
DTree.fit(X_train.values,Y_train)
Y_predict=DTree.predict(X_test.values)
#I think this accuracy score is enough
from sklearn.metrics import f1_score
f1_score(Y_predict,Y_test)
