#importing the necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import r2_score as r2

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder
#reading the StudentsPerformance.csv file and viewing it

student = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")

student.head()
#Identifying the columns present in the dataset

student.columns
#This displays general information about the dataset with informations like the column names their data types 

#and the count of non-null values for every column.

student.info()
#checking if there is any column which contains null values

student.isna().sum()
#this will help in knowing the number of categories present in each categorical variable

student.select_dtypes('object').nunique()
#to find out the various categories present in the different categorical variable

print("Categories in 'gender' variable: ",end=" ")

print(student['gender'].unique())

print("Categories in 'race/ethnicity' variable: ",end=" ")

print(student['race/ethnicity'].unique())

print("Categories in 'parental level of education' variable: ",end=" ")

print(student['parental level of education'].unique())

print("Categories in 'lunch' variable: ",end=" ")

print(student['lunch'].unique())

print("Categories in 'test preparation course' variable: ",end=" ")

print(student['test preparation course'].unique())
#This displays information about the quantitive/numerical columns, information like count, mean, standard deviation, minimum value, maximum value 

#and the quartiles are displayed 

student.describe()
#Total score = math score + reading score + writing score

student['Total Score']=student['math score']+student['reading score']+student['writing score']
#Criterion for getting a passing grade

def result(TS,MS,WS,RS ):

    if(TS>120 and MS>40 and WS>40 and RS>40):

        return 'P'

    else:

        return 'F'

    
student['Pass/Fail']=student.apply(lambda x: result(x['Total Score'],x['math score'],x['writing score'],x['reading score']),axis = 1 )
student.head()
#Displays the number of students passed and failed according to the passing criterion

student['Pass/Fail'].value_counts()
plt.pie(student['Pass/Fail'].value_counts(),labels=['Pass','Fail'],autopct='%1.1f%%')

plt.title('Percentage of students Passed/Failed')
sns.countplot(student['Pass/Fail'])

plt.title('Bar-plot representing the count of students passed/failed')
# this displays the number of male and female students in the class

student['gender'].value_counts()
#to find out the percentage of female students passed

print("Percentage of female students passed: {0:.2f}%"

    .format((student[(student['gender']=='female') & (student['Pass/Fail']=='P')].shape[0]/student[student['gender']=='female'].shape[0])*100))



#to find out the percentage of male students passed

print("Percentage of male students passed: {0:.2f}%"

    .format((student[(student['gender']=='male') & (student['Pass/Fail']=='P')].shape[0]/student[student['gender']=='male'].shape[0])*100))
sns.countplot(student['Pass/Fail'],hue = student['gender'])

plt.ylabel('Number of students')
fig,ax = plt.subplots(3,1, figsize = (5,10))

sns.barplot(x=student['gender'],y=student['math score'], ax=ax[0], linewidth=2.5)

sns.barplot(x=student['gender'],y=student['reading score'], ax=ax[1],linewidth=2.5)

sns.barplot(x=student['gender'],y=student['writing score'], ax=ax[2],linewidth=2.5)

plt.tight_layout()
fig,ax = plt.subplots(3,1, figsize = (5,10))

sns.boxplot(x=student['gender'],y=student['math score'],ax=ax[0])

sns.boxplot(x=student['gender'],y=student['reading score'],ax=ax[1])

sns.boxplot(x=student['gender'],y=student['writing score'],ax=ax[2])

plt.tight_layout()
#number of students belonging to each race/ethnic group

student['race/ethnicity'].value_counts()
#number of students passed across the race/ethnic groups

print("The number of students passed across various race/ethnic group : ")

print(student['race/ethnicity'].loc[student['Pass/Fail']=='P'].value_counts())

sns.countplot(student['race/ethnicity'].loc[student['Pass/Fail']=='P'])

plt.xticks(rotation = 45)
sns.countplot(student['race/ethnicity'],hue=student['Pass/Fail'])

plt.ylabel('Number of students')
#to find out the percentage of students passed with the race/ethnicity  as 'group A'

print("Percentage of students passed with the race/ethnicity  as 'group A': {0:.2f}%"

    .format((student[(student['race/ethnicity']=='group A') & (student['Pass/Fail']=='P')].shape[0]/student[student['race/ethnicity']=='group A'].shape[0])*100))



#to find out the percentage of students passed with the race/ethnicity  as 'group B'

print("Percentage of students passed with the race/ethnicity  as 'group B': {0:.2f}%"

    .format((student[(student['race/ethnicity']=='group B') & (student['Pass/Fail']=='P')].shape[0]/student[student['race/ethnicity']=='group B'].shape[0])*100))



#to find out the percentage of students passed with the race/ethnicity  as 'group C'

print("Percentage of students passed with the race/ethnicity  as 'group C': {0:.2f}%"

    .format((student[(student['race/ethnicity']=='group C') & (student['Pass/Fail']=='P')].shape[0]/student[student['race/ethnicity']=='group C'].shape[0])*100))



#to find out the percentage of students passed with the race/ethnicity  as 'group D'

print("Percentage of students passed with the race/ethnicity  as 'group D': {0:.2f}%"

    .format((student[(student['race/ethnicity']=='group D') & (student['Pass/Fail']=='P')].shape[0]/student[student['race/ethnicity']=='group D'].shape[0])*100))



#to find out the percentage of students passed with the race/ethnicity  as 'group E'

print("Percentage of students passed with the race/ethnicity  as 'group E': {0:.2f}%"

    .format((student[(student['race/ethnicity']=='group E') & (student['Pass/Fail']=='P')].shape[0]/student[student['race/ethnicity']=='group E'].shape[0])*100))

fig, ax = plt.subplots(3,1, figsize=(8,12))

sns.boxplot(x=student['race/ethnicity'],y=student['math score'],ax=ax[0])

sns.boxplot(x=student['race/ethnicity'],y=student['reading score'],ax=ax[1])

sns.boxplot(x=student['race/ethnicity'],y=student['writing score'],ax=ax[2])

plt.tight_layout()
#number of students having parents with various edication level

student['parental level of education'].value_counts()
#number of students passed across the parental levels of education 

print("The number of students passed across the different parental levels of education: ")

print(student['parental level of education'].loc[student['Pass/Fail']=='P'].value_counts())

sns.countplot(student['parental level of education'].loc[student['Pass/Fail']=='P'])

plt.xticks(rotation = 45)
#to find out the percentage of students passed with the parental level of education as 'some college'

print("Percentage of students passed with the parental level of education as 'some college': {0:.2f}%"

    .format((student[(student['parental level of education']=='some college') & (student['Pass/Fail']=='P')].shape[0]/student[student['parental level of education']=='some college'].shape[0])*100))



#to find out the percentage of students passed with the parental level of education as 'associate's degree'

print("Percentage of students passed with the parental level of education as 'associate's degree': {0:.2f}%"

    .format((student[(student['parental level of education']=="associate's degree") & (student['Pass/Fail']=='P')].shape[0]/student[student['parental level of education']=="associate's degree"].shape[0])*100))



#to find out the percentage of students passed with the parental level of education as 'high school'

print("Percentage of students passed with the parental level of education as 'high school': {0:.2f}%"

    .format((student[(student['parental level of education']=="high school") & (student['Pass/Fail']=='P')].shape[0]/student[student['parental level of education']=="high school"].shape[0])*100))



#to find out the percentage of students passed with the parental level of education as 'some high school'

print("Percentage of students passed with the parental level of education as 'some high school': {0:.2f}%"

    .format((student[(student['parental level of education']=="some high school") & (student['Pass/Fail']=='P')].shape[0]/student[student['parental level of education']=="some high school"].shape[0])*100))



#to find out the percentage of students passed with the parental level of education as 'bachelor's degree'

print("Percentage of students passed with the parental level of education as 'bachelor's degree': {0:.2f}%"

    .format((student[(student['parental level of education']=="bachelor's degree") & (student['Pass/Fail']=='P')].shape[0]/student[student['parental level of education']=="bachelor's degree"].shape[0])*100))



#to find out the percentage of students passed with the parental level of education as 'master's degree'

print("Percentage of students passed with the parental level of education as 'master's degree': {0:.2f}%"

    .format((student[(student['parental level of education']=="master's degree") & (student['Pass/Fail']=='P')].shape[0]/student[student['parental level of education']=="master's degree"].shape[0])*100))
plt.figure(figsize= (10,8))

sns.countplot(student['parental level of education'],hue=student['Pass/Fail'])

plt.xticks(rotation=90)

plt.ylabel('Number of students')
plt.figure(figsize=(10,5))

plt.title("Total Score across parental level of education of students")

sns.barplot(x=student['parental level of education'],y=student['Total Score'])
#number of students having 'standard' lunch vs. number of students having 'free/reduced' lunch

student['lunch'].value_counts()
#number of students passed across the type of lunch 

student['lunch'].loc[student['Pass/Fail']=='P'].value_counts()
sns.countplot(student['lunch'],hue=student['Pass/Fail'])
#to find out the percentage of students passed with the lunch type as 'standard'

print("Percentage of students passed with the lunch type as 'standard': {0:.2f}%"

    .format((student[(student['lunch']=='standard') & (student['Pass/Fail']=='P')].shape[0]/student[student['lunch']=='standard'].shape[0])*100))



#to find out the percentage of students passed with the lunch type as 'free/reduced'

print("Percentage of students passed with the lunch type as 'free/reduced': {0:.2f}%"

    .format((student[(student['lunch']=="free/reduced") & (student['Pass/Fail']=='P')].shape[0]/student[student['lunch']=="free/reduced"].shape[0])*100))

plt.figure(figsize=(5,5))

plt.title("Total Score across the type of lunch of the students")

sns.barplot(x=student['lunch'],y=student['Total Score'],hue=student['gender'])
#number of students who completed the 'Test preparation course' vs. the students who didn't complete the course

student['test preparation course'].value_counts()
#number of students passed across the status of completion of the test preparation course 

print("The number of students passed across the status of completion of the test preparation course:")

print(student['test preparation course'].loc[student['Pass/Fail']=='P'].value_counts())

#to find out the percentage of students passed with the test preparation course status as 'none'

print("Percentage of students passed with the test preparation course status as 'none': {0:.2f}%"

    .format((student[(student['test preparation course']=='none') & (student['Pass/Fail']=='P')].shape[0]/student[student['test preparation course']=='none'].shape[0])*100))



#to find out the percentage of students passed with the test preparation course status as 'completed'

print("Percentage of students passed with the test preparation course status as 'completed': {0:.2f}%"

    .format((student[(student['test preparation course']=="completed") & (student['Pass/Fail']=='P')].shape[0]/student[student['test preparation course']=="completed"].shape[0])*100))

plt.figure(figsize=(5,5))

sns.barplot(x=student['test preparation course'],y=student['Total Score'])

plt.title("Total Score across the status of test prep course")

plt.xlabel('Status of Test Prep Course')
fig, ax = plt.subplots(3,1, figsize=(8,12))

sns.regplot(x=student['reading score'],y=student['writing score'],ax = ax[0])

sns.regplot(x=student['reading score'],y=student['math score'],ax = ax[1])

sns.regplot(x=student['writing score'],y=student['math score'],ax=ax[2])

plt.tight_layout()
sns.heatmap(student.corr(), cmap ="Reds")

plt.xticks(rotation=90)
X=student[['gender','race/ethnicity','parental level of education','lunch','test preparation course']]

X.head()
X_category = student[['gender','race/ethnicity','parental level of education','lunch','test preparation course']]
# Applying one-hot encoding to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_OH = pd.DataFrame(OH_encoder.fit_transform(X_category))

X_OH.index = X.index #One-hot encoding removes the index so it's necessary to put them back

X_OH.head()
#collecting the total score of the students from the dataset

y=student['Pass/Fail']

y.head()
lb=LabelEncoder()

y=lb.fit_transform(y)
# Divide data into training and validation subsets

X_train, X_valid, y_train, y_valid = train_test_split(X_OH, y, train_size=0.8, test_size=0.2,random_state=0)
model = RandomForestRegressor()

model.fit(X_train,y_train)
#model predicting

preds=model.predict(X_valid)#predictions made by the model
preds= np.where(preds<0.4,0,1)
preds
y_valid
#Calculating the Mean Absolute Error value

mae(y_valid,preds)
scores = -1 * cross_val_score(model, X_OH, y,cv=5,scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
from sklearn.metrics import confusion_matrix



# creating a confusion matrix

cm = confusion_matrix(y_valid, preds)



# printing the confusion matrix

plt.rcParams['figure.figsize'] = (8, 8)

sns.heatmap(cm, annot = True, cmap = 'Reds')

plt.show()