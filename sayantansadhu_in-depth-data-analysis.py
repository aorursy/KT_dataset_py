# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
data.head( )
data.info()
data['total score'] = data['math score'] + data['reading score'] + data['writing score']
data['race/ethnicity'].value_counts()
sns.countplot(data['race/ethnicity'])

plt.title('Number of people belonging to each race/ethnicity')
ethnic_group = data.groupby(['race/ethnicity'])

ethnic_group['total score'].max()
ethnic_group_mean = ethnic_group['total score'].mean()

print(ethnic_group_mean)
import matplotlib.pyplot as plt

sns.barplot(x = ['Group A','Group B','Group C','Group D','Group E'], y = ethnic_group_mean)

plt.title('Performance of different ethnic group based on their total score')
ethnic_group_math = ethnic_group['math score'].mean()

print(ethnic_group['math score'].max())

print(ethnic_group_math)
sns.barplot(x = ['Group A','Group B','Group C','Group D','Group E'], y = ethnic_group_math)

plt.title('Performance of different ethnic group based on their math score')
print(ethnic_group['reading score'].max())

print(ethnic_group['reading score'].mean())
sns.barplot(x = ['Group A','Group B','Group C','Group D','Group E'], y = ethnic_group['reading score'].mean())

plt.title('Performance of different ethnic group based on their reading score')
print(ethnic_group['writing score'].max())

print(ethnic_group['writing score'].mean())
sns.barplot(x = ['Group A','Group B','Group C','Group D','Group E'], y = ethnic_group['writing score'].mean())

plt.title('Performance of different ethnic group based on their writing score')
data.loc[data['total score'] <= 120].count()
data.loc[data['total score'] <= 120]
data.loc[data['total score'] <= 120]['race/ethnicity'].value_counts()

sns.countplot(data.loc[data['total score'] <= 120]['race/ethnicity'])

plt.title('Number of students failed from each race/ethnicity')
print('Percentage of total student failed ', round(32/1000*100,2),'%')

print('Percentage of student failed for race A', round(3/89*100,2),'%')

print('Percentage of student failed for race B', round(8/190*100,2),'%')

print('Percentage of student failed for race C', round(10/319*100,2),'%')

print('Percentage of student failed for race D', round(8/262*100,2),'%')

print('Percentage of student failed for race E', round(2/140*100,2),'%')
sns.countplot(data['gender'])

print(data['gender'].value_counts())
sns.countplot(data['gender'], hue = data['race/ethnicity'])
gender_group = data.groupby('gender')
gender_group_mean = gender_group['total score'].mean()

print(gender_group_mean)
sns.barplot(y=gender_group_mean,x = ['female','male'])

plt.title('Mean score of male and female')
data.loc[data['total score'] <= 120]['gender'].value_counts()
print('percentage of females who failed the test', round(17/518*100,2),'%')

print('percentage of males who failed the test', round(15/482*100,2),'%')
data['parental level of education'].value_counts()
plt.figure(figsize=(10,5))

sns.countplot(data['parental level of education'])
parent_education_group = data.groupby(['parental level of education'])
parent_education_group_mean = parent_education_group['total score'].mean()

print(parent_education_group_mean )
plt.figure(figsize=(10,5))

sns.barplot(x = ['Associates degree','bachelor degree','high school','masters degree','some college','some high school'], y = parent_education_group['total score'].mean())

plt.title('Performance based on the qualification of parents ')
data.loc[data['total score'] <= 120]['parental level of education'].value_counts()
plt.figure(figsize=(10,5))

sns.countplot(data.loc[data['total score'] <= 120]['parental level of education'])
data['test preparation course'].value_counts()
course_completion_group = data.groupby('test preparation course')

course_completion_group_mean = course_completion_group ['total score'].mean()

print(course_completion_group_mean)
sns.barplot(x = ['completed','none'], y = course_completion_group ['total score'].mean())

plt.title('Performance based on whether a student has completed the test preparation course or not')
data.loc[data['total score'] <= 120]['test preparation course'].value_counts()
sns.countplot(data.loc[data['total score'] <= 120]['test preparation course'])