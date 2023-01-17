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
import matplotlib.pyplot as plt

import seaborn as sns



import os

import warnings
emp_attr_df = pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
emp_attr_df.head()
emp_attr_df.info()
emp_attr_df.describe()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(emp_attr_df.corr(),annot=True,linewidths=0.5,linecolor="green",fmt=".1f",ax=ax)
g = sns.pairplot(emp_attr_df, vars=["MonthlyIncome", "MonthlyRate"],hue="Department",height=5)
labels=emp_attr_df.EducationField.value_counts().index

colors=["olive","orange","hotpink","slateblue","y","lime"]

sizes=emp_attr_df.EducationField.value_counts().values

plt.figure(figsize=(7,7))

plt.pie(sizes,labels=labels,colors=colors,autopct="%1.1f%%")

plt.title("Education Field Counts",color="saddlebrown",fontsize=15)
f,ax = plt.subplots(figsize = (15,10))

sns.boxplot(x="Department",y="MonthlyIncome",hue="MaritalStatus",data=emp_attr_df,palette="Paired")
f,ax = plt.subplots(figsize = (15,10))

sns.boxplot(x="Gender",y="Age",hue="BusinessTravel",data=emp_attr_df,palette="hls")

ax.legend(loc='upper center',frameon = True)
f,ax = plt.subplots(figsize = (15,15))

sns.swarmplot(x="JobRole",y="HourlyRate",hue="Attrition",data=emp_attr_df,palette="hls")

plt.xticks(rotation=90)
plt.subplots(figsize=(15,10))

sns.swarmplot(x="JobRole",y="MonthlyIncome",hue="EducationField",data=emp_attr_df, palette="hls")

plt.xticks(rotation=90)
plt.subplots(figsize=(15,5))

sns.countplot(emp_attr_df.Age, hue=emp_attr_df.Attrition)
plt.subplots(figsize=(15,5))

sns.countplot(emp_attr_df.TotalWorkingYears, hue=emp_attr_df.Attrition)
plt.subplots(figsize=(15,5))

sns.countplot(emp_attr_df.YearsWithCurrManager, hue=emp_attr_df.Attrition)
plt.subplots(figsize=(15,5))

sns.countplot(emp_attr_df.YearsSinceLastPromotion, hue=emp_attr_df.Attrition)
sns.countplot(emp_attr_df.Education, hue=emp_attr_df.Attrition)
sns.countplot(emp_attr_df.NumCompaniesWorked, hue=emp_attr_df.Attrition)
plt.subplots(figsize=(15,5))

sns.countplot(emp_attr_df.DistanceFromHome, hue=emp_attr_df.Attrition)
age_df=pd.DataFrame(emp_attr_df.groupby("Age")[["MonthlyIncome","Education","JobLevel","JobInvolvement","PerformanceRating","JobSatisfaction","EnvironmentSatisfaction","RelationshipSatisfaction","WorkLifeBalance","DailyRate","MonthlyRate"]].mean())

age_df["Count"]=emp_attr_df.Age.value_counts(dropna=False)

age_df.reset_index(level=0, inplace=True)

age_df.head()
plt.figure(figsize=(15,10))

ax=sns.barplot(x=age_df.Age,y=age_df.Count)

plt.xticks(rotation=90)

plt.xlabel("Age")

plt.ylabel("Counts")

plt.title("Age Counts")
plt.figure(figsize=(15,10))

ax=sns.barplot(x=age_df.Age,y=age_df.MonthlyIncome,palette = sns.cubehelix_palette(len(age_df.index)))

plt.xticks(rotation=90)

plt.xlabel("Age")

plt.ylabel("Monthly Income")

plt.title("Monthly Income According to Age")
income_df = pd.DataFrame(emp_attr_df.groupby("JobRole").MonthlyIncome.mean().sort_values(ascending=False))

income_df
plt.figure(figsize=(15,10))

ax=sns.barplot(x=income_df.index,y=income_df.MonthlyIncome)

plt.xticks(rotation=90)

plt.xlabel("Job Roles")

plt.ylabel("Monthly Income")

plt.title("Job Roles with Monthly Income")
jobrole_df = pd.DataFrame(emp_attr_df.groupby("JobRole")["PercentSalaryHike","YearsAtCompany","TotalWorkingYears","YearsInCurrentRole","WorkLifeBalance"].mean())

jobrole_df
f,ax = plt.subplots(figsize = (9,10))

sns.barplot(x=jobrole_df.PercentSalaryHike,y=jobrole_df.index,color='green',alpha = 0.5,label='Percent Salary Hike' )

sns.barplot(x=jobrole_df.TotalWorkingYears,y=jobrole_df.index,color='blue',alpha = 0.7,label='Average Working Years')

sns.barplot(x=jobrole_df.YearsAtCompany,y=jobrole_df.index,color='cyan',alpha = 0.6,label='Years At Company')

sns.barplot(x=jobrole_df.YearsInCurrentRole,y=jobrole_df.index,color='yellow',alpha = 0.6,label='Years In Current Role')

sns.barplot(x=jobrole_df.WorkLifeBalance,y=jobrole_df.index,color='red',alpha = 0.6,label='Work-Life Balance')



ax.legend(loc='lower right',frameon = True)     

ax.set(xlabel='Values', ylabel='Job Roles',title = "Job Roles with Different Features")
agenorm_df = age_df.apply(lambda x: x/max(x))

agenorm_df.head()
ageb=emp_attr_df.Age.value_counts().index

f,ax=plt.subplots(figsize=(15,15))

sns.pointplot(y=agenorm_df.WorkLifeBalance,x=ageb,color="purple",alpha=0.8)

sns.pointplot(y=agenorm_df.JobSatisfaction,x=ageb,color="sandybrown",alpha=0.8)

plt.text(5,0.65,"Worklife Balance",color="purple",fontsize=15,style="italic")

plt.text(5,0.63,"Job Satisfaction",color="sandybrown",fontsize=15,style="italic")

plt.xlabel("Age",fontsize=15,color="darkred")

plt.ylabel("Values",fontsize=15,color="darkred")

plt.title("Worklife Balance VS Job Satisfaction",fontsize=15,color="darkred")

plt.grid()
g=sns.jointplot(agenorm_df.JobInvolvement,agenorm_df.MonthlyIncome,kind="reg",height=10)
g=sns.jointplot(agenorm_df.RelationshipSatisfaction,agenorm_df.EnvironmentSatisfaction,kind="hex",height=10)
g=sns.jointplot("MonthlyRate","DailyRate",data=agenorm_df,height=10,ratio=3,color="tomato")
f,ax = plt.subplots(figsize = (15,10))

sns.kdeplot(agenorm_df.PerformanceRating,agenorm_df.Education,shade=False,cut=1)
agejoblevel_df = pd.DataFrame(emp_attr_df.groupby("TotalWorkingYears")["MonthlyIncome"].mean())

agejoblevel_df.reset_index(level=0,inplace=True)

agejoblevel_df.TotalWorkingYears = agejoblevel_df.TotalWorkingYears/max(agejoblevel_df.TotalWorkingYears)

agejoblevel_df.MonthlyIncome = agejoblevel_df.MonthlyIncome/max(agejoblevel_df.MonthlyIncome)
pal=sns.cubehelix_palette(2,rot=-.5,dark=.3)

sns.violinplot(data=agejoblevel_df,palette=pal,inner="points")
sns.pairplot(agejoblevel_df)
sns.lmplot(x="TotalWorkingYears",y="MonthlyIncome",data=agejoblevel_df)