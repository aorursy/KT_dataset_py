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

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

data.head()
data.info()
data.describe()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidths=0.5,linecolor="green",fmt=".1f",ax=ax)

plt.show()
age=pd.DataFrame(data.groupby("Age")[["MonthlyIncome","Education","JobLevel","JobInvolvement","PerformanceRating","JobSatisfaction","EnvironmentSatisfaction","RelationshipSatisfaction","WorkLifeBalance","DailyRate","MonthlyRate"]].mean())

age["Count"]=data.Age.value_counts(dropna=False)

age.reset_index(level=0, inplace=True)

age.head()
plt.figure(figsize=(15,10))

ax=sns.barplot(x=age.Age,y=age.Count)

plt.xticks(rotation=90)

plt.xlabel("Age")

plt.ylabel("Counts")

plt.title("Age Counts")

plt.show()
plt.figure(figsize=(15,10))

ax=sns.barplot(x=age.Age,y=age.MonthlyIncome,palette = sns.cubehelix_palette(len(age.index)))

plt.xticks(rotation=90)

plt.xlabel("Age")

plt.ylabel("Monthly Income")

plt.title("Monthly Income According to Age")

plt.show()
income=pd.DataFrame(data.groupby("JobRole").MonthlyIncome.mean().sort_values(ascending=False))
plt.figure(figsize=(15,10))

ax=sns.barplot(x=income.index,y=income.MonthlyIncome)

plt.xticks(rotation=90)

plt.xlabel("Job Roles")

plt.ylabel("Monthly Income")

plt.title("Job Roles with Monthly Income")

plt.show()
jobrole=pd.DataFrame(data.groupby("JobRole")["PercentSalaryHike","YearsAtCompany","TotalWorkingYears","YearsInCurrentRole","WorkLifeBalance"].mean())

jobrole
f,ax = plt.subplots(figsize = (9,10))

sns.barplot(x=jobrole.PercentSalaryHike,y=jobrole.index,color='green',alpha = 0.5,label='Percent Salary Hike' )

sns.barplot(x=jobrole.TotalWorkingYears,y=jobrole.index,color='blue',alpha = 0.7,label='Average Working Years')

sns.barplot(x=jobrole.YearsAtCompany,y=jobrole.index,color='cyan',alpha = 0.6,label='Years At Company')

sns.barplot(x=jobrole.YearsInCurrentRole,y=jobrole.index,color='yellow',alpha = 0.6,label='Years In Current Role')

sns.barplot(x=jobrole.WorkLifeBalance,y=jobrole.index,color='red',alpha = 0.6,label='Work-Life Balance')



ax.legend(loc='lower right',frameon = True)     

ax.set(xlabel='Values', ylabel='Job Roles',title = "Job Roles with Different Features")

plt.show()
agenorm=age.apply(lambda x: x/max(x))

agenorm.head()
ageb=data.Age.value_counts().index

f,ax=plt.subplots(figsize=(15,15))

sns.pointplot(y=agenorm.WorkLifeBalance,x=ageb,color="purple",alpha=0.8)

sns.pointplot(y=agenorm.JobSatisfaction,x=ageb,color="sandybrown",alpha=0.8)

plt.text(5,0.65,"Worklife Balance",color="purple",fontsize=15,style="italic")

plt.text(5,0.63,"Job Satisfaction",color="sandybrown",fontsize=15,style="italic")

plt.xlabel("Age",fontsize=15,color="darkred")

plt.ylabel("Values",fontsize=15,color="darkred")

plt.title("Worklife Balance VS Job Satisfaction",fontsize=15,color="darkred")

plt.grid()
g = sns.pairplot(data, vars=["MonthlyIncome", "MonthlyRate"],hue="Department",size=5)
labels=data.EducationField.value_counts().index

colors=["olive","orange","hotpink","slateblue","y","lime"]

#explode=[0,0,0,0,0,0]

sizes=data.EducationField.value_counts().values

plt.figure(figsize=(7,7))

plt.pie(sizes,labels=labels,colors=colors,autopct="%1.1f%%")

plt.title("Education Field Counts",color="saddlebrown",fontsize=15)
g=sns.jointplot(agenorm.JobInvolvement,agenorm.MonthlyIncome,kind="reg",size=10)
g=sns.jointplot(agenorm.RelationshipSatisfaction,agenorm.EnvironmentSatisfaction,kind="hex",size=10)
g=sns.jointplot("MonthlyRate","DailyRate",data=agenorm,size=10,ratio=3,color="tomato")
agenorm.head()
f,ax = plt.subplots(figsize = (15,10))

sns.kdeplot(agenorm.PerformanceRating,agenorm.Education,shade=False,cut=1)
f,ax = plt.subplots(figsize = (15,10))

sns.boxplot(x="Department",y="MonthlyIncome",hue="MaritalStatus",data=data,palette="Paired")
f,ax = plt.subplots(figsize = (15,10))

sns.boxplot(x="Gender",y="Age",hue="BusinessTravel",data=data,palette="hls")
f,ax = plt.subplots(figsize = (15,15))

sns.swarmplot(x="JobRole",y="HourlyRate",hue="Attrition",data=data,palette="hls")

plt.xticks(rotation=90)
plt.subplots(figsize=(15,10))

sns.swarmplot(x="JobRole",y="MonthlyIncome",hue="EducationField",data=data,palette="hls")

plt.xticks(rotation=90)
agejoblevel=pd.DataFrame(data.groupby("TotalWorkingYears")["MonthlyIncome"].mean())

agejoblevel.reset_index(level=0,inplace=True)
agejoblevel.TotalWorkingYears=agejoblevel.TotalWorkingYears/max(agejoblevel.TotalWorkingYears)

agejoblevel.MonthlyIncome=agejoblevel.MonthlyIncome/max(agejoblevel.MonthlyIncome)
pal=sns.cubehelix_palette(2,rot=-.5,dark=.3)

sns.violinplot(data=agejoblevel,palette=pal,inner="points")

plt.show()
sns.pairplot(agejoblevel)
sns.lmplot(x="TotalWorkingYears",y="MonthlyIncome",data=agejoblevel)

plt.show()
plt.subplots(figsize=(15,5))

sns.countplot(data.TotalWorkingYears)
sns.countplot(data.Education)
sns.countplot(data.NumCompaniesWorked)
sns.countplot(data.DistanceFromHome)