#for basic operation

import pandas as pd

import numpy as np



#for visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#reading the data

csv_path = r"../input/students-performance-in-exams/StudentsPerformance.csv"

df =pd.read_csv(csv_path)
#looking for top of the data

df.head(3)
df.shape
#checking the missing values

df.isnull().sum()
#changing column names for convenience

df.columns = ['gender','race','parents_education','food','course','maths','read','write']
#adding new columns 

df['total'] = (df['maths']+ df['read']+ df['write'])

df['percentage'] = df['total']/3

df['percentage'] =df['percentage'].astype(float).round(2)

df.sample()
#checking the types of variables

df.dtypes
df.describe()
df.describe(include='object')
def distribution(x):

    sns.distplot(x)
distribution(df['maths'])
distribution(df['read'])
distribution(df['write'])
distribution(df['percentage'])
def cat_plot(x):

    plt.figure(figsize=(25,10))

    plt.subplot(2,3,1)

    sns.violinplot(y='gender',x=x,data=df)

    plt.subplot(2,3,2)

    sns.violinplot(y='race',x=x,data=df)

    plt.subplot(2,3,4)

    sns.violinplot(y='parents_education',x=x,data=df)

    plt.subplot(2,3,3)

    sns.violinplot(y='food',x=x,data=df)

    plt.subplot(2,3,5)

    sns.violinplot(y='course',x=x,data=df)
#viuualize the distribution of categorical variables in maths score

cat_plot('maths')
#viuualize the distribution of categorical variables in reading score 

cat_plot('read')
#viuualize the distribution of categorical variables in writing score 

cat_plot('write')
cat_plot('percentage')
#checking individual object

df['gender'].unique()
#visualizing the percentage of gender

df['gender'].value_counts().plot.pie(labels =['Female','Male'],colors =['b','c'],fontsize =12, autopct='%.1f',figsize=(4,4))
#seperate the columns using groupby

df1=df.groupby('gender')[['maths','read','write']]
#getting average scores based on gender

df1.mean()
#visualizing the above code

df1.mean().plot.bar(figsize = (6,4),color=['b','c','k'],alpha=0.6)

plt.title('Average scores on male vs female',fontsize=16,alpha=1.0)

plt.ylabel('Marks',fontsize=14,alpha=1.0)

plt.xlabel('Gender',fontsize=14,alpha=1.0)

plt.xticks(rotation = 0)

plt.legend(bbox_to_anchor=(1,1),loc = 'upper left')

plt.show()
#getting mean for percentage based on gender

avg = df.groupby('gender')['percentage']

avg.mean()
#visualizing the average of percentage

avg.mean().plot.barh(figsize = (6,3),color ='m',alpha=0.4)

plt.title('Average based on Gender',fontsize=16,alpha=1.0)

plt.ylabel('Gender',fontsize=14,alpha=1.0)

plt.xlabel('Marks',fontsize=14,alpha=1.0)

plt.show()
df1.mean().mean().plot.bar(figsize = (6,4),color='g',alpha=0.8)

plt.title("Average of Marks",color ='k',fontsize=18,alpha =1.0)

plt.ylabel('Marks',fontsize =16,alpha = 1.0)

plt.xlabel('Subjects',fontsize =16,alpha = 1.0)

plt.xticks(rotation = 45)

plt.show()
#visualizing parents education levels

df['parents_education'].value_counts().plot.pie(autopct='%.0f')

plt.title("Parents education level",fontsize=16,color='k',alpha =1.0)

plt.ylabel(" ")

plt.legend(bbox_to_anchor=(1.3,1),loc="upper left")

plt.show()
#comparison of scores based on gender and parents education

parent = df.groupby(['gender','parents_education'])['percentage'].mean()
#visualizing the comparison of scores based on gender and parents education

parent.unstack().plot.bar(figsize=(8,5),alpha = 1.0)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.title('Scores based on Gender & Parents education',fontsize= 20,alpha =1.0,color ='k')

plt.xticks(rotation =0)

plt.xlabel('Gender',fontsize = 16,alpha = 1.0)

plt.show()