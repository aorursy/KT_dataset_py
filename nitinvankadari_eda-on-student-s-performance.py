import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
studata = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
studata.head()
studata.info()
studata['total score']=(studata['math score'] + studata['reading score'] + studata['writing score'])/3
studata.head()
studata.describe()
sns.heatmap(studata.corr())
plt.figure(figsize=(20, 10))

sns.countplot(studata['math score'], palette = 'Set1').set_title('Math Score')
plt.figure(figsize=(20, 10))

sns.countplot(studata['reading score'], palette = 'Set2').set_title('Math Score')
plt.figure(figsize=(20, 10))

sns.countplot(studata['writing score'], palette = 'Set3').set_title('Math Score')
studata.groupby(['gender']).mean()
studata['gender'].unique()
plt.figure(figsize=(5, 5))

plt.title('Gender')

studata['gender'].value_counts().plot.pie(autopct="%1.1f%%")
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.boxplot(x = 'gender', y = 'math score', data = studata).set_title('GENDER VS MATH SCORE')



plt.subplot(1,3,2)

sns.boxplot(x = 'gender', y = 'reading score', data = studata).set_title('GENDER VS READING SCORE')



plt.subplot(1,3,3)

sns.boxplot(x = 'gender', y = 'writing score', data = studata).set_title('GENDER VS WRITING SCORE')
sns.barplot(x='total score',y='gender',data=studata).set_title('GENDER VS TOTAL SCORE')
studata.groupby(['lunch']).mean()
studata['lunch'].unique()
plt.figure(figsize=(5, 5))

plt.title('Lunch')

studata['lunch'].value_counts().plot.pie(autopct="%1.1f%%")
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.violinplot(x = 'lunch' , y = 'math score' , data = studata).set_title('LUNCH VS MATH SCORE')



plt.subplot(1,3,2)

sns.violinplot(x ='lunch', y = 'reading score' , data = studata).set_title('LUNCH VS READING SCORE')



plt.subplot(1,3,3)

sns.violinplot(x = 'lunch', y = 'writing score' , data = studata).set_title('LUNCH VS WRITING SCORE')
sns.barplot(x='total score',y='lunch',data=studata).set_title('LUNCH VS TOTAL SCORE')
studata.groupby(['parental level of education']).mean()
studata['parental level of education'].unique()
plt.figure(figsize=(5, 5))

plt.title('Parental level of education')

studata['parental level of education'].value_counts().plot.pie(autopct="%1.1f%%")
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.barplot(x = 'parental level of education' , y = 'math score' , data = studata).set_title('PARENTAL LEVEL OF EDUCATION VS MATH SCORE')

plt.xticks(rotation = 90)



plt.subplot(1,3,2)

sns.barplot(x ='parental level of education', y = 'reading score' , data = studata).set_title('PARENTAL LEVEL OF EDUCATION VS READING SCORE')

plt.xticks(rotation = 90)



plt.subplot(1,3,3)

sns.barplot(x = 'parental level of education', y = 'writing score' , data = studata).set_title('PARENTAL LEVEL OF EDUCATION VS WRITING SCORE')

plt.xticks(rotation = 90)



plt.tight_layout()
sns.barplot(x='total score',y='parental level of education',data=studata).set_title('PARENTAL LEVEL OF EDUCATION VS WRITING SCORE VS TOTAL SCORE')
studata.groupby(['race/ethnicity']).mean()
studata['race/ethnicity'].unique()
plt.figure(figsize=(5, 5))

plt.title('Race/Ethnicity')

studata['race/ethnicity'].value_counts().plot.pie(autopct="%1.1f%%")
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.stripplot(x = 'race/ethnicity' , y = 'math score' , data = studata).set_title('RACE/ETHNICITY VS MATH SCORE')

plt.xticks(rotation = 90)



plt.subplot(1,3,2)

sns.stripplot(x ='race/ethnicity', y = 'reading score' , data = studata).set_title('RACE/ETHNICITY VS READING SCORE')

plt.xticks(rotation = 90)



plt.subplot(1,3,3)

sns.stripplot(x = 'race/ethnicity', y = 'writing score' , data = studata).set_title('RACE/ETHNICITY VS WRITING SCORE')

plt.xticks(rotation = 90)
sns.barplot(x='total score',y='race/ethnicity',data=studata).set_title('RACE/ETHNICITY VS TOTAL SCORE')
studata.groupby(['test preparation course']).mean()
studata['test preparation course'].unique()
plt.figure(figsize=(5, 5))

plt.title('Test preparation course')

studata['test preparation course'].value_counts().plot.pie(autopct="%1.1f%%")
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.boxplot(x = 'test preparation course' , y = 'math score' , data = studata).set_title('TEST PREPARATION COURSE VS MATH SCORE')



plt.subplot(1,3,2)

sns.boxplot(x ='test preparation course', y = 'reading score' , data = studata).set_title('TEST PREPARATION COURSE VS READING SCORE')



plt.subplot(1,3,3)

sns.boxplot(x = 'test preparation course', y = 'writing score' , data = studata).set_title('TEST PREPARATION COURSE VS WRITING SCORE')
sns.barplot(x='total score',y='test preparation course',data=studata).set_title('TEST PREPARATION COURSE VS TOTAL SCORE')