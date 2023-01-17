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



# Any results you write to the current directory are saved as output.
%matplotlib inline

plt.style.use('fivethirtyeight')
#Reading the Data

df = pd.read_csv("../input/StudentsPerformance.csv")

df.head()
df.columns
df.shape
df.info()
#Checking for Nulls

df.isnull().sum()
df.describe()
#Adding a column of total score in the dataframe

df['total score'] = df['math score'] + df['reading score'] + df['writing score']

df.head()
#Checking data types of the columns

df.dtypes
#Correlation among numeric columns

df.corr()
sns.heatmap(df.corr())

plt.title('Covariance Plot')
#Scatter plot of scores in different subjects to visualize correlation among them

plt.figure(figsize = (20,8))



plt.subplot(1,3,1)

plt.scatter(df['math score'],df['reading score'])

plt.title('Math scores vs Reading scores')

plt.xlabel('Math score')

plt.ylabel('Reading score')



plt.subplot(1,3,2)

plt.scatter(df['reading score'],df['writing score'])

plt.title('Reading scores vs Writing scores')

plt.xlabel('Reading score')

plt.ylabel('Writing score')



plt.subplot(1,3,3)

plt.scatter(df['writing score'],df['math score'])

plt.title('Writing scores vs Math scores')

plt.xlabel('Writing score')

plt.ylabel('Math score')



plt.show()
#Same plot as above but with regression line using Seaborn

plt.figure(figsize = (20,8))



plt.subplot(1,3,1)

sns.regplot(x = 'math score', y = 'reading score',data = df)

plt.title('Math scores vs Reading scores')

plt.xlabel('Math score')

plt.ylabel('Reading score')



plt.subplot(1,3,2)

sns.regplot(x = 'reading score', y = 'writing score',data = df)

plt.title('Reading scores vs Writing scores')

plt.xlabel('Reading score')

plt.ylabel('Writing score')



plt.subplot(1,3,3)

sns.regplot(x = 'writing score', y = 'math score',data = df)

plt.title('Writing scores vs Math scores')

plt.xlabel('Writing score')

plt.ylabel('Math score')



plt.show()
#Scatter plot between individiual subject score and total score with using Seaborn

plt.figure(figsize = (20,8))



plt.subplot(1,3,1)

sns.regplot(x = 'math score', y = 'total score',data = df)

plt.title('Math scores vs Total scores')

plt.xlabel('Math score')

plt.ylabel('Total score')



plt.subplot(1,3,2)

sns.regplot(x = 'reading score', y = 'total score',data = df)

plt.title('Reading scores vs Total scores')

plt.xlabel('Reading score')

plt.ylabel('Total score')



plt.subplot(1,3,3)

sns.regplot(x = 'writing score', y = 'total score',data = df)

plt.title('Writing scores vs Total scores')

plt.xlabel('Writing score')

plt.ylabel('Total score')



plt.show()
#Number of Students against Scores in all the 3 subjects

plt.hist([df['math score'],df['reading score'],df['writing score']], color=['red', 'yellow', 'blue'])

plt.title('Number of Students against Scores')

plt.xlabel('Score')

plt.ylabel('Number of Students')

plt.legend(['Math', 'Reading', 'Writing'])

plt.show()
y = ['Math','Reading','Writing']

width = [df['math score'].mean(),df['reading score'].mean(),df['writing score'].mean()]



plt.figure(figsize = (12,2))

plt.barh(y = y, 

         width = width)

plt.title('Average Scores')

plt.xlabel('Average Score')

plt.ylabel('Subjects')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(round(v,2)), color='blue', va='center', fontweight='bold')

plt.show()
df_gender = df.groupby('gender')
#Number of Females and Males

y = df_gender['gender'].count().keys()

width = df_gender['gender'].count()

plt.figure(figsize = (12,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of Females and Males')

plt.xlabel('Count')

plt.ylabel('Gender')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
#Average scores of Females and Males

y = df_gender['total score'].mean().keys()

width = df_gender['total score'].mean()

plt.figure(figsize = (12,2))

plt.barh(y = y, 

         width = width)

plt.title('Av score of Female and Males')

plt.xlabel('Av. total score out of 300')

plt.ylabel('Gender')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(round(v,2)), color='blue', va='center', fontweight='bold')

plt.show()
sns.boxplot(x="gender", y="total score", data=df)
sns.swarmplot(x='gender',y='total score',data=df)

sns.violinplot(x='gender',y='total score',data=df, inner=None,color='lightgray')
#Average scores in individual subjects

x = df_gender['gender'].count().keys()



plt.figure(figsize=(12,4))



plt.subplot(1,3,1)

height = df_gender['math score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Math Score')

plt.xlabel('Gender')

plt.ylabel('Av. Math Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')



plt.subplot(1,3,2)

height = df_gender['reading score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Reading Score')

plt.xlabel('Gender')

plt.ylabel('Av. Reading Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')



plt.subplot(1,3,3)

height = df_gender['writing score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Writing Score')

plt.xlabel('Gender')

plt.ylabel('Av. Writing Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha = 'center', fontweight='bold')



plt.show()
df_race = df.groupby('race/ethnicity')
#Count of studensts belonging to different race/ethnicity groups

y = df_race['race/ethnicity'].count().keys()

width = df_race['race/ethnicity'].count()

plt.figure(figsize = (12,4))

plt.barh(y = y, 

         width = width)

plt.title('No. of Students of Different Race/Ethnicity Groups')

plt.xlabel('Count')

plt.ylabel('Race/Ethnicity')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
#Average score of students in different race/ethnicity groups

y = df_race['total score'].mean().keys()

width = df_race['total score'].mean()

plt.figure(figsize = (12,4))

plt.barh(y = y, 

         width = width)

plt.title('Mean Scores of Students of Different Race/Ethnicity Groups')

plt.xlabel('Mean score')

plt.ylabel('Race/Ehtnicity')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(round(v,2)), color='blue', va='center', fontweight='bold')

plt.show()
plt.figure(figsize = (12,4))

sns.boxplot(x="race/ethnicity", y="total score", data=df)
plt.figure(figsize=(12, 6))

sns.swarmplot(x='race/ethnicity',y='total score',data=df, hue = 'gender')

sns.violinplot(x='race/ethnicity',y='total score',data=df, inner=None,color='lightgray')
#Average scores in individual subject based on Race/Ethnicity

x = df_race['total score'].mean().keys()



plt.figure(figsize=(20,5))



plt.subplot(1,3,1)

height = df_race['math score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Math Score')

plt.xlabel('race/ethnicity')

plt.ylabel('Av. Math Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')



plt.subplot(1,3,2)

height = df_race['reading score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Reading Score')

plt.xlabel('race/ethnicity')

plt.ylabel('Av. Reading Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')



plt.subplot(1,3,3)

height = df_race['writing score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Writing Score')

plt.xlabel('race/ethnicity')

plt.ylabel('Av. Writing Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha = 'center', fontweight='bold')



plt.show()
df_parental = df.groupby('parental level of education')
#Counting students based on the parental level of education

y = df_parental['parental level of education'].count().keys()

width = df_parental['parental level of education'].count()

plt.figure(figsize = (12,4))

plt.barh(y = y, 

         width = width)

plt.title('No. of Students based on parental level of education')

plt.xlabel('Count')

plt.ylabel('Parental level of education')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
#Mean score of students based on the parental level of education

y = df_parental['total score'].mean().keys()

width = df_parental['total score'].mean()

plt.figure(figsize = (12,4))

plt.barh(y = y, 

         width = width)

plt.title('Mean score of Students based on parental level of education')

plt.xlabel('Mean total score')

plt.ylabel('Parental levelof education')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(round(v,2)), color='blue', va='center', fontweight='bold')

plt.show()
plt.figure(figsize = (12,4))

sns.boxplot(x="parental level of education", y="total score", data=df)
plt.figure(figsize=(13, 6))

sns.swarmplot(x='parental level of education',y='total score',data=df, hue = 'gender')

sns.violinplot(x='parental level of education',y='total score',data=df, inner=None,color='lightgray')
#Mean scores of students in individual subjects based on parental level of education 

x = df_parental['total score'].mean().keys()



plt.figure(figsize=(20,18))



plt.subplot(3,1,1)

height = df_parental['math score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Math Score')

plt.xlabel('parental level if education')

plt.ylabel('Av. Math Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')



plt.subplot(3,1,2)

height = df_parental['reading score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Reading Score')

plt.xlabel('parental level if education')

plt.ylabel('Av. Reading Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')



plt.subplot(3,1,3)

height = df_parental['writing score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Writing Score')

plt.xlabel('parental level if education')

plt.ylabel('Av. Writing Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha = 'center', fontweight='bold')



plt.show()
df_lunch = df.groupby('lunch')
# Counting students according to lunch type

y = df_lunch['lunch'].count().keys()

width = df_lunch['lunch'].count()

plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of Students acc. to lunch type')

plt.xlabel('Count')

plt.ylabel('Lunch')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
#Mean score of students according to lunch type

y = df_lunch['total score'].mean().keys()

width = df_lunch['total score'].mean()

plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('Mean total score of Students in Different Lunch categories')

plt.xlabel('Mean total score')

plt.ylabel('Lunch')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(round(v,2)), color='blue', va='center', fontweight='bold')

plt.show()
plt.figure(figsize = (8,4))

sns.boxplot(x="lunch", y="total score", data=df)
plt.figure(figsize=(10, 4))

sns.swarmplot(x='lunch',y='total score',data=df, hue = 'gender')

sns.violinplot(x='lunch',y='total score',data=df, inner=None,color='lightgray')
#Mean scores of students in individual subjects based on lunch

x = df_lunch['total score'].mean().keys()



plt.figure(figsize=(12,4))



plt.subplot(1,3,1)

height = df_lunch['math score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Math Score')

plt.xlabel('Gender')

plt.ylabel('Av. Math Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')



plt.subplot(1,3,2)

height = df_lunch['reading score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Reading Score')

plt.xlabel('Gender')

plt.ylabel('Av. Reading Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')



plt.subplot(1,3,3)

height = df_lunch['writing score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Writing Score')

plt.xlabel('Gender')

plt.ylabel('Av. Writing Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha = 'center', fontweight='bold')



plt.show()
df_test = df.groupby('test preparation course')
#Count of students based on test preparation course

y = df_test['test preparation course'].count().keys()

width = df_test['test preparation course'].count()

plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of Students with and without test preparation course')

plt.xlabel('Count')

plt.ylabel('Test preparation course')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
#Mean scores of students based on test preparation course

y = df_test['total score'].mean().keys()

width = df_test['total score'].mean()

plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('Mean total score of Students with and without a test preparation course')

plt.xlabel('Mean total score')

plt.ylabel('Test preparation course')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(round(v,2)), color='blue', va='center', fontweight='bold')

plt.show()
plt.figure(figsize = (8,4))

sns.boxplot(x="test preparation course", y="total score", data=df)
plt.figure(figsize=(8, 4))

sns.swarmplot(x='test preparation course',y='total score',data=df, hue = 'gender')

sns.violinplot(x='test preparation course',y='total score',data=df, inner=None,color='lightgray')
#Individual scores based on test preparation course

x = df_test['total score'].mean().keys()



plt.figure(figsize=(12,4))



plt.subplot(1,3,1)

height = df_test['math score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Math Score')

plt.xlabel('Test preparation course')

plt.ylabel('Av. Math Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')



plt.subplot(1,3,2)

height = df_test['reading score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Reading Score')

plt.xlabel('Test preparation course')

plt.ylabel('Av. Reading Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')



plt.subplot(1,3,3)

height = df_test['writing score'].mean()

plt.bar(x = x, 

         height = height)

plt.title('Av. Writing Score')

plt.xlabel('Test preparation course')

plt.ylabel('Av. Writing Score')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(round(v,2)), color='blue', ha = 'center', fontweight='bold')



plt.show()
#Toppers

df_topper = df[(df['math score'] >= 90) & (df['reading score'] >= 90) & (df['writing score'] >= 90)]
df_topper['gender'].count()
df_topper.sort_values('total score',ascending=False).head()
#Students who have failed in all three subjects(less than 40 score)

df_fail_all = df[(df['math score'] < 40) & (df['reading score'] < 40) & (df['writing score'] < 40)]

df_fail_all.head()
y = df['gender'].unique().tolist()

x = [0]*len(y)

for i in range(len(y)):

    x[i] += df_fail_all[df_fail_all['gender'] == y[i]]['gender'].count()

width = x



plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of students who failed in all 3 subjects based on gender')

plt.xlabel('Count')

plt.ylabel('Gender')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
y = df['race/ethnicity'].unique().tolist()

x = [0]*len(y)

for i in range(len(y)):

    x[i] += df_fail_all[df_fail_all['race/ethnicity'] == y[i]]['gender'].count()

width = x



plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of Students who failed in all 3 subjects in Different Race/Ethnicity Groups')

plt.xlabel('Count')

plt.ylabel('Race')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
y = df['parental level of education'].unique().tolist()

x = [0]*len(y)

for i in range(len(y)):

    x[i] += df_fail_all[df_fail_all['parental level of education'] == y[i]]['gender'].count()

width = x

plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of Students who failed all subjects based on parental education')

plt.xlabel('Count')

plt.ylabel('parental level of education')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
y = df['lunch'].unique().tolist()

x = [0]*len(y)

for i in range(len(y)):

    x[i] += df_fail_all[df_fail_all['lunch'] == y[i]]['gender'].count()

width = x



plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of Students who failed all subjects based on lunch')

plt.xlabel('Count')

plt.ylabel('Lunch')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
y = df['test preparation course'].unique().tolist()

x = [0]*len(y)

for i in range(len(y)):

    x[i] += df_fail_all[df_fail_all['test preparation course'] == y[i]]['gender'].count()

width = x



plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of students who failed all subjects based on test preparation course')

plt.xlabel('Count')

plt.ylabel('Test preparation course')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
#Students who failed in atleast 1 subject (less than 40 score in atleast 1 subject)

df_fail = df[(df['math score'] < 40) | (df['reading score'] < 40) | (df['writing score'] < 40)]

df_fail.head()
width = [df_fail[df_fail['math score'] < 40]['gender'].count(), df_fail[df_fail['reading score']< 40]['gender'] .count(), 

    df_fail[df_fail['writing score'] < 40]['gender'] .count()]



y = ['math','reading','writing']



plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of students who failed in individual subjects')

plt.xlabel('Count')

plt.ylabel('Subject')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
y = df['gender'].unique().tolist()

x = [0]*len(y)

for i in range(len(y)):

    x[i] += df_fail[df_fail['gender'] == y[i]]['gender'].count()

width = x



plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of students who failed atleast 1 subject based on gender')

plt.xlabel('Count')

plt.ylabel('gender')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
#Number of students who failed in individual subjects based on gender

plt.figure(figsize = (25,10))

x = ['female','male']



plt.subplot(1,3,1)



height = [df_fail[(df_fail['gender']=='female') & (df_fail['math score'] < 40)]['gender'].count(), 

        df_fail[(df_fail['gender']=='male') & (df_fail['math score'] < 40)]['gender'].count()]

plt.bar(x = x, 

         height = height)

plt.title('No. of students who failed in maths based on gender')

plt.ylabel('count')

plt.xlabel('gender')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(v), color='blue', ha='center', fontweight='bold')



plt.subplot(1,3,2)

height = [df_fail[(df_fail['gender']=='female') & (df_fail['reading score'] < 40)]['gender'].count(), 

        df_fail[(df_fail['gender']=='male') & (df_fail['reading score'] < 40)]['gender'].count()]



plt.bar(x = x, 

         height = height)

plt.title('No. of students who failed in reading based on gender')

plt.ylabel('count')

plt.xlabel('gender')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(v), color='blue', ha='center', fontweight='bold')

    

plt.subplot(1,3,3)

height = [df_fail[(df_fail['gender']=='female') & (df_fail['writing score'] < 40)]['gender'].count(), 

        df_fail[(df_fail['gender']=='male') & (df_fail['writing score'] < 40)]['gender'].count()]

plt.bar(x = x, 

         height = height)

plt.title('No. of students who failed in writing based on gender')

plt.ylabel('Count')

plt.xlabel('gender')

for i,v in enumerate(height):

    plt.text(i, v, " "+str(v), color='blue', ha='center', fontweight='bold')

    

plt.show()
y = df['race/ethnicity'].unique().tolist()

x = [0]*len(y)

for i in range(len(y)):

    x[i] += df_fail[df_fail['race/ethnicity'] == y[i]]['gender'].count()

width = x



plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of students who failed atleast 1 subject based on race/ethnicity')

plt.xlabel('Count')

plt.ylabel('Race/Ethnicity')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
y = df['parental level of education'].unique().tolist()

x = [0]*len(y)

for i in range(len(y)):

    x[i] += df_fail[df_fail['parental level of education'] == y[i]]['gender'].count()

width = x



plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of students who failed atleast 1 subject based on parental level of education')

plt.xlabel('Count')

plt.ylabel('parental level of education')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
y = df['lunch'].unique().tolist()

x = [0]*len(y)

for i in range(len(y)):

    x[i] += df_fail[df_fail['lunch'] == y[i]]['gender'].count()

width = x



plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of students who failed atleast 1 subject  based acc. to lunch')

plt.xlabel('Count')

plt.ylabel('Lunch')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()
y = df['test preparation course'].unique().tolist()

x = [0]*len(y)

for i in range(len(y)):

    x[i] += df_fail[df_fail['test preparation course'] == y[i]]['gender'].count()

width = x



plt.figure(figsize = (10,2))

plt.barh(y = y, 

         width = width)

plt.title('No. of students who failed atleast 1 subject based on test preparation course')

plt.xlabel('Count')

plt.ylabel('Test preparation course')

for i,v in enumerate(width):

    plt.text(v, i, " "+str(v), color='blue', va='center', fontweight='bold')

plt.show()