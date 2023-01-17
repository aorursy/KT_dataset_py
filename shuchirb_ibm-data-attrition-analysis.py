# Call libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings    

import os

from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score  

from sklearn.tree import DecisionTreeClassifier

from pandas import read_csv
warnings.filterwarnings("ignore")    # Ignore warnings

IBMData=pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
IBMData.head()
IBMData.shape
pd.isnull(IBMData).sum()
for i in range(IBMData.shape[1]):

    print (IBMData.columns.values[i],":",np.unique(IBMData[IBMData.columns.values[i]]),"\n")

IBMData=IBMData.drop(['EmployeeCount','Over18','StandardHours','DailyRate'],axis=1)
IBMData=IBMData.replace({'Attrition':{'No':0,'Yes':1}})

IBMData=IBMData.replace({'Gender':{'Male':0,'Female':1}})

IBMData=IBMData.replace({'BusinessTravel':{'Non-Travel':0,'Travel_Rarely':1,'Travel_Frequently':2}})

IBMData=IBMData.replace({'Department':{'Human Resources':0,'Research & Development':1,'Sales':2}})

IBMData=IBMData.replace({'EducationField':{'Human Resources':0,'Life Sciences':1,'Marketing':2,'Medical':3,'Technical Degree':4,'Other':5}})

IBMData=IBMData.replace({'JobRole':{'Human Resources':0,'Healthcare Representative':1,'Laboratory Technician':2,'Manager':3,'Manufacturing Director':4,'Research Director':5,'Research Scientist':6,'Sales Executive':7,'Sales Representative':8}})

IBMData=IBMData.replace({'MaritalStatus':{'Single':0,'Married':1,'Divorced':2}}) 

IBMData=IBMData.replace({'OverTime':{'No':0,'Yes':1}})

IBMData['MonthlyIncome']=IBMData['MonthlyIncome']-min(IBMData['MonthlyIncome'])/(max(IBMData['MonthlyIncome'])-min(IBMData['MonthlyIncome']))

IBMData['MonthlyRate']=IBMData['MonthlyRate']-min(IBMData['MonthlyRate'])/(max(IBMData['MonthlyRate'])-min(IBMData['MonthlyRate']))

IBMData.shape
IBMData.head()
row=8

col=3

fig,ax = plt.subplots(row,col, figsize=(15,20))  # 'ax' has references to all the four axes

x=0

y=0

for i in range(IBMData.shape[1]):

    if(i!=1):

        sns.distplot(IBMData[IBMData.columns.values[i]], ax = ax[x,y])  # Plot on 1st axes 

        sns.boxplot(IBMData['Attrition'],IBMData[IBMData.columns.values[i]], ax = ax[x+1,y])  # Plot on 1st axes 

    y=y+1

    if(x<row-2 and y==col):

        x=x+2

        y=0

    if(x==row-2 and y==col):

        break

plt.show()

row=8

col=3

fig,ax = plt.subplots(row,col, figsize=(15,20))  # 'ax' has references to all the four axes

x=0

y=0

for i in range(12,IBMData.shape[1]):

    if(i!=1):

        sns.distplot(IBMData[IBMData.columns.values[i]], ax = ax[x,y])  # Plot on 1st axes 

        sns.boxplot(IBMData['Attrition'],IBMData[IBMData.columns.values[i]], ax = ax[x+1,y])  # Plot on 1st axes 

    y=y+1

    if(x<row-2 and y==col):

        x=x+2

        y=0

    if(x==row-2 and y==col):

        break

plt.show()
row=6

col=3

fig,ax = plt.subplots(row,col, figsize=(15,25))  # 'ax' has references to all the four axes

x=0

y=0

for i in range(24,IBMData.shape[1]):

    if(i!=1):

        sns.distplot(IBMData[IBMData.columns.values[i]], ax = ax[x,y])  # Plot on 1st axes 

        sns.boxplot(IBMData['Attrition'],IBMData[IBMData.columns.values[i]], ax = ax[x+1,y])  # Plot on 1st axes 

    y=y+1

    if(x<row-2 and y==col):

        x=x+2

        y=0

    if(x==row-2 and y==col):

        break

plt.show()
newIBMData = IBMData.drop(['Age','Department','Education','EducationField','JobInvolvement','MaritalStatus','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating', 'RelationshipSatisfaction','StockOptionLevel','TrainingTimesLastYear','WorkLifeBalance'],axis=1)       
df_y=newIBMData.loc[:,"Attrition"]

df_x=newIBMData.drop(["Attrition"],axis=1)

x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=.2,random_state=4)

model=DecisionTreeClassifier()

fittedModel=model.fit(x_train, y_train)

predictions=fittedModel.predict(x_test)

confusion=confusion_matrix(y_test,predictions)

accuracy=accuracy_score(y_test,predictions)

print('Accuracy:',accuracy)

accuracy