#Import packages
import os
import pandas as pd #To work with dataset
import numpy as np #Math library
import seaborn as sns #Graph library that use matplot in background
import matplotlib.pyplot as plt #to plot some parameters in seaborn
%matplotlib inline
os.getcwd()
#Change the working directory
#os.chdir("H:\\python\\PYTHON\\data visulaization")
#Load the data set
hr=pd.read_csv("../input/eda-for-hr-data-set/hr.csv")
hr.head()
#Looking the Type of Data
hr.dtypes
#look at the Shape of data
hr.shape
#Looking the missing values,type and shape of the data
hr.info()
#hr.dtypes
#Look at the unique values of Gender varible in the data set
hr['Gender'].unique()
#Look at the first 5 observations of the data set
hr.head()
#Summary Statistics
hr.describe()
#View sum of the missing values of each variables
hr.isnull().sum()
hr["Attrition"].value_counts()
#Target variable distribution
#import seaborn as sns
sns.countplot("Attrition", data=hr,saturation=0.68)
#import matplotlib.pyplot as plt #to plot some parameters in seaborn
#%matplotlib inline
attritioncount=hr['Attrition'].value_counts()
#print(attritioncount)
labels=['No','Yes']
colors=['gold','lightskyblue']
explode=(0.2,0)

plt.pie(attritioncount,explode=explode,labels=labels,colors=colors,
        autopct='%1.1f%%',shadow=True,startangle=45)
plt.show()
pd.value_counts(hr["Attrition"])
count_classes = pd.value_counts(hr['Attrition'], sort = True).sort_index()
print(count_classes)
#hr['Attrition'].value_counts()
pd.crosstab(hr.Gender,hr.Attrition)
print(pd.crosstab(hr.Gender,hr.Attrition,normalize='index'))
pd.crosstab(hr.Gender,hr.Attrition).plot(kind='bar')
plt.title('Attrition with respect to Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency of Attrition')
plt.show()
hr.Department.value_counts()
pd.crosstab(hr.Department,hr.Attrition).plot(kind='bar')
#Breakdown of Attribution data with respect to EducationField
print(pd.crosstab(hr['EducationField'],hr['Attrition']))
pd.crosstab(hr['EducationField'],hr['Attrition']).plot(kind='bar')

plt.title('Attrition with respect to EducationField')
plt.xlabel('EducationField')
plt.ylabel('Frequency of Attrition')
#plt.show()
#Work experience
pd.crosstab(hr.Workex,hr.Attrition).plot(kind='barh',color=('blue','green'),figsize=(10,15))
pd.crosstab(hr.Performance_Rating,hr.Attrition)
pd.crosstab(hr.OverTime,hr.Attrition,normalize=True)
pd.crosstab(hr.OverTime,hr.Attrition).plot(kind='bar',color=('blue','green'),figsize=(5,5))
pd.crosstab(hr.Attrition,hr.JobSatisfaction)
hr['MaritalStatus'].value_counts()
#print(pd.crosstab(hr.Attrition,hr.MaritalStatus)
pd.crosstab(hr.Attrition,hr.MaritalStatus,normalize='index')

#pd.crosstab(hr.Attrition,hr.YearsSinceLastPromotion,normalize='index')
pd.crosstab(hr.Attrition,hr.YearsSinceLastPromotion)
#pd.crosstab(hr.Attrition,hr.MaritalStatus,normalize='index')
pd.crosstab(hr.Attrition,hr.JobRole)
pd.crosstab(hr.Salaryhike,hr.Attrition).plot(kind='bar',color=('blue','green'),figsize=(15,5))
#Leavers by business travel
pd.crosstab(hr.BusinessTravel,hr.Attrition).plot(kind='bar')
plt.title('Attrition with respect to BusinessTravel')
plt.xlabel('BusinessTravel')
plt.ylabel('Frequency of Attrition')
#Let us get a sense of the numbers across these two classes:
hr.groupby('Attrition').mean()
#Create age groups 
bins = [18, 30, 40, 50, 60]
labels = ['18-30', '30-40', '40-50', '50-60']
hr['agegroup'] = pd.cut(hr.Age, bins, labels = labels,include_lowest = True)
pd.crosstab(hr.agegroup,hr.Attrition)
#cm = sns.light_palette("green", as_cmap=True)
#pd.crosstab(hr['agegroup'], hr['Attrition']).style.background_gradient(cmap = cm)
colnames = ['Education','JobInvolvement', 'JobLevel','JobSatisfaction', 'Performance_Rating', 'OverTime', 'WorkLifeBalance']
for c in colnames:
    hr[c] = hr[c].astype('category')
corrmat = data.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, square=True,annot=True)
# convert to numeric for mean
hr.Attrition_numeric = hr.Attrition
hr.loc[hr.Attrition == 'Yes','Attrition_numeric'] = 1
hr.loc[hr.Attrition == 'No','Attrition_numeric'] = 0

plt.figure(figsize=(4,5))
plt.title("Leavers by Gender")
sns.barplot(x = 'Gender', y = 'Attrition_numeric', data=hr)
plt.figure(figsize=(12,8))
sns.barplot(x = 'OverTime', y = 'Attrition_numeric', data=hr)

