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

import pandas as pd

import seaborn as sns

import numpy as np

import pandas as pd

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

df = pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

df.head()
#Looking for null values 

df.isnull().sum()
#Checking the website: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset

#We are dropping the following columns due to unnecessary information: EmployeeCount, EmployeeNumber, StandarHours, Over18

df=df.drop(columns=['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18','PerformanceRating'])

df.head(3)
#What tyoe of variable are looking at

df.info()
categorical=df.select_dtypes(include=['object'])

categorical.head()
#Collinearity Analysis

plt.figure(figsize=(20,20))

sns.heatmap(df.corr(),annot=True);
df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
#df.Attrition.groupby()

print("Yes: ",df[df["Attrition"]=='Yes'].count()['Attrition'])

print("No: ",df[df["Attrition"]=='No'].count()['Attrition'])
df_attrition_department= df[["Attrition","Department"]].groupby ("Department").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="Department", y="percentage", hue="Attrition", data=df_attrition_department) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_BusinessTravel= df[["Attrition","BusinessTravel"]].groupby ("BusinessTravel").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="BusinessTravel", y="percentage", hue="Attrition", data=df_attrition_BusinessTravel) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_Gender= df[["Attrition","Gender"]].groupby ("Gender").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="Gender", y="percentage", hue="Attrition", data=df_attrition_Gender) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_EducationField= df[["Attrition","EducationField"]].groupby ("EducationField").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="EducationField", y="percentage", hue="Attrition", data=df_attrition_EducationField) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_JobRole= df[["Attrition","JobRole"]].groupby ("JobRole").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="JobRole", y="percentage", hue="Attrition", data=df_attrition_JobRole) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_MaritalStatus= df[["Attrition","MaritalStatus"]].groupby ("MaritalStatus").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="MaritalStatus", y="percentage", hue="Attrition", data=df_attrition_MaritalStatus) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_OverTime= df[["Attrition","OverTime"]].groupby ("OverTime").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="OverTime", y="percentage", hue="Attrition", data=df_attrition_OverTime) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_department= df[["Attrition","Education"]].groupby ("Education").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="Education", y="percentage", hue="Attrition", data=df_attrition_department) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_department= df[["Attrition","EnvironmentSatisfaction"]].groupby ("EnvironmentSatisfaction").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="EnvironmentSatisfaction", y="percentage", hue="Attrition", data=df_attrition_department) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_department= df[["Attrition","JobInvolvement"]].groupby ("JobInvolvement").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="JobInvolvement", y="percentage", hue="Attrition", data=df_attrition_department) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_department= df[["Attrition","JobLevel"]].groupby ("JobLevel").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="JobLevel", y="percentage", hue="Attrition", data=df_attrition_department) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_department= df[["Attrition","JobSatisfaction"]].groupby ("JobSatisfaction").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="JobSatisfaction", y="percentage", hue="Attrition", data=df_attrition_department) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
    

df_attrition_department= df[["Attrition","RelationshipSatisfaction"]].groupby ("RelationshipSatisfaction").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="RelationshipSatisfaction", y="percentage", hue="Attrition", data=df_attrition_department) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_department= df[["Attrition","WorkLifeBalance"]].groupby ("WorkLifeBalance").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="WorkLifeBalance", y="percentage", hue="Attrition", data=df_attrition_department) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show
df_attrition_department= df[["Attrition","StockOptionLevel"]].groupby ("StockOptionLevel").Attrition.value_counts(normalize=True).rename('percentage').reset_index()

p=sns.barplot(x="StockOptionLevel", y="percentage", hue="Attrition", data=df_attrition_department) 

plt.setp(p.get_xticklabels(), rotation=90)

plt.show


dfmean=df.groupby('Attrition').mean()

#df.columns

dfmean.drop(columns=['Education','EnvironmentSatisfaction','JobInvolvement','JobLevel','JobSatisfaction','RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager'])
dfmean.reset_index()

print(dfmean)

#Combining relationship between relevant variables

g = sns.catplot(x="BusinessTravel", hue="Attrition", col="MaritalStatus",data=df, kind="count")#,height=4, aspect=.7);
#Combining relationship between relevant variables

g = sns.catplot(x="EnvironmentSatisfaction", hue="Attrition", col="OverTime",data=df, kind="count")#,height=4, aspect=.7);