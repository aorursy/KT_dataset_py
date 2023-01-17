# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(r'/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
df.head(5)
#Categorical Varibales  : gender, race/ethnicity, parental level of education, lunch,'test preparation course'

#Quantitative Variables : math score,reading score,writing score
df.shape
df.dtypes
df.describe(include ='all')
df.head(5).columns
df.isnull().sum()
df.rename(columns={'parental level of education':'parentEDU','test preparation course':'testPEPScore'},inplace =True)
#Lets see the frequeny distribution of the categorical variables



#Most of the students are female

x = df[['gender']].value_counts()

x = x/x.sum()

print(x)

print('')

#Most numbe of students from Group C enthenicity followed by Group D

x = df[['race/ethnicity']].value_counts()

x = x/x.sum()

print(x)



print('')



#Most of the parents of education of some college or above

x = df[['parentEDU']].value_counts()

x = x/x.sum()

print(x)



print('')

#Most students prefer stanadrd lunch 

x = df[['lunch']].value_counts()

x = x/x.sum()

print(x)



print('')



x = df[['testPEPScore']].value_counts()

x = x/x.sum()

print(x)
#Lets visualise through some plots

cat_variables = ['gender', 'race/ethnicity', 'parentEDU', 'lunch','testPEPScore']

for col in cat_variables:

    plt.figure(figsize = (10,4))

    sns.barplot(data = df, x= col,y=df.index)

    plt.show()

#Comments : All the variables that we have considered are all

#           unimodel(having only one peak),are forming a bell shape and left skewed

quan_variables = ['math score','reading score','writing score']

for col in quan_variables:

    plt.figure(figsize = (10,4))

    sns.distplot(df[col],kde = False,hist= True)

    plt.show()
df[quan_variables].plot(kind= 'box')
#On thing we can see that median (50%) and mean are very close 

#even though there are few outliers are visible in box and hist plot mean is not getting affedcted.

df[quan_variables].describe()
#We will using the two way table or contingency table explore categorical variables

pd.crosstab(df['parentEDU'],df['race/ethnicity']).apply(lambda x:x/x.sum(),axis = 1)
pd.crosstab(df['gender'],df['testPEPScore']).apply(lambda x:x/x.sum(),axis = 1)
pd.crosstab(df['race/ethnicity'],df['lunch']).apply(lambda x:x/x.sum(),axis = 1)
df[['reading score', 'writing score','math score']].corr()
sns.jointplot(x='writing score',y='reading score',data=df,kind ='kde')
#Lets check we find any relation between catgeorical variables and scores of students visualy.

#Comment : Male student have higher scores than females only in mathematics

fig = plt.figure(figsize =(18,4))

ax1= fig.add_subplot(131)

sns.boxplot(data =df ,x='gender',y='math score',orient = 'v',ax=ax1)

ax2= fig.add_subplot(132)

sns.boxplot(data =df ,x='gender',y='reading score',orient = 'v',ax=ax2)

ax3= fig.add_subplot(133)

sns.boxplot(data =df ,x='gender',y='writing score',orient = 'v',ax=ax3)
#Here I have ploted box plots by stratifying on the gender to get more inghits.But it can be

#without stratifying also.



#Comment : We cam clearly see that average score of students whose parents hold some kind of degree

#          is higher then compared to others in all subjects.

fig = plt.figure(figsize =(10 ,12))

ax1= fig.add_subplot(311)

sns.boxplot(data =df ,x='parentEDU',y='math score',orient = 'v',hue= 'gender',ax=ax1)

ax2= fig.add_subplot(312)

sns.boxplot(data =df ,x='parentEDU',y='reading score',orient = 'v',hue= 'gender',ax=ax2)

ax3= fig.add_subplot(313)

sns.boxplot(data =df ,x='parentEDU',y='writing score',orient = 'v',ax=ax3,hue= 'gender')
# We are not able get any kind info between ethinicity and scores

# but students belonging to Group A has on avergae low scores

fig = plt.figure(figsize =(10 ,12))

ax1= fig.add_subplot(311)

sns.boxplot(data =df ,x='race/ethnicity',y='math score',orient = 'v',ax=ax1,hue= 'gender')

ax2= fig.add_subplot(312)

sns.boxplot(data =df ,x='race/ethnicity',y='reading score',orient = 'v',ax=ax2,hue= 'gender')

ax3= fig.add_subplot(313)

sns.boxplot(data =df ,x='race/ethnicity',y='writing score',orient = 'v',ax=ax3,hue= 'gender')
#Student have opted for standard lunch have on average 

# higher scores than the students who opted for free and standard 

fig = plt.figure(figsize =(6 ,14))

ax1= fig.add_subplot(311)

sns.boxplot(data =df ,x='lunch',y='math score',orient = 'v',ax=ax1,hue= 'gender')

ax2= fig.add_subplot(312)

sns.boxplot(data =df ,x='lunch',y='reading score',orient = 'v',ax=ax2,hue= 'gender')

ax3= fig.add_subplot(313)

sns.boxplot(data =df ,x='lunch',y='writing score',orient = 'v',ax=ax3,hue= 'gender')
#This is obvious students who have completed Test preparation course have higher scores.

fig = plt.figure(figsize =(6 ,14))

ax1= fig.add_subplot(311)

sns.boxplot(data =df ,x='testPEPScore',y='math score',orient = 'v',ax=ax1,hue= 'gender')

ax2= fig.add_subplot(312)

sns.boxplot(data =df ,x='testPEPScore',y='reading score',orient = 'v',ax=ax2,hue= 'gender')

ax3= fig.add_subplot(313)

sns.boxplot(data =df ,x='testPEPScore',y='writing score',orient = 'v',ax=ax3,hue= 'gender')