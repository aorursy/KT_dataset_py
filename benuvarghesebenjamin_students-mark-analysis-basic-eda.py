import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
data.head()
data.columns = ['Gender', 'Race', 'Parent Degree', 'Lunch', 'Course', 'Math Score', 'Reading Score', 'Writing Score'] 
data.head()
data.isnull().sum()
data['Total Score']=(data['Math Score'] + data['Reading Score'] + data['Writing Score'])/3
data.head()
data.groupby(['Gender']).mean()
sns.barplot(x='Total Score',y='Gender',data=data).set_title('GENDER VS TOTAL SCORE')
sns.barplot(x='Math Score',y='Gender',data=data).set_title('GENDER VS MATH SCORE')
data['Math Score'].mode()
data['Math Score'].mean()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.boxplot(x = 'Gender', y = 'Math Score', data = data)



plt.subplot(1,3,2)

sns.boxplot(x = 'Gender', y = 'Reading Score', data = data)



plt.subplot(1,3,3)

sns.boxplot(x = 'Gender', y = 'Writing Score', data = data)
sns.barplot(x='Lunch',y='Total Score',data=data)
plt.figure(figsize=(13,4))



plt.subplot(1,3,1)

sns.barplot(x = "Parent Degree" , y="Reading Score" , data=data)

plt.xticks(rotation = 90)

plt.title("Reading Scores")



plt.subplot(1,3,2)

sns.barplot(x = "Parent Degree" , y="Writing Score" , data=data)

plt.xticks(rotation=90)

plt.title("Writing Scores")



plt.subplot(1,3,3)

sns.barplot(x = "Parent Degree" , y="Math Score" , data=data)

plt.xticks(rotation=90)

plt.title("Math Scores")



plt.tight_layout()

plt.show()
sns.barplot(x='Race',y='Total Score',data=data)
df1=data.sort_values('Total Score',ascending=False)
df1.head()
fig,ax=plt.subplots()

sns.countplot(x='Parent Degree',data=data,palette='spring')

plt.tight_layout()

fig.autofmt_xdate()