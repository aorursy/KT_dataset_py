# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
df.info()
df.describe()
df.shape
df.isnull().sum() #checks if there are any missing values
plt.rcParams['figure.figsize'] = (20, 10)

sns.countplot(df['math score'], palette = 'dark')

plt.title('Math Score',fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (20, 10)

sns.countplot(df['reading score'], palette = 'Set3')

plt.title('Reading Score',fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (20, 10)

sns.countplot(df['writing score'], palette = 'prism')

plt.title('Writing Score',fontsize = 20)

plt.show()
plt.figure(figsize=(15,5))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,

                      wspace=0.5, hspace=0.2)

plt.subplot(141)

plt.title('Math Scores')

sns.violinplot(y='math score',data=df,color='m',linewidth=2)

plt.subplot(142)

plt.title('Reading Scores')

sns.violinplot(y='reading score',data=df,color='g',linewidth=2)

plt.subplot(143)

plt.title('Writing Scores')

sns.violinplot(y='writing score',data=df,color='r',linewidth=2)

plt.show()
plt.figure(figsize=(20,10))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,

                      wspace=0.5, hspace=0.2)

plt.subplot(141)

plt.title('Gender',fontsize = 20)

df['gender'].value_counts().plot.pie(autopct="%1.1f%%")



plt.subplot(142)

plt.title('Ethinicity',fontsize = 20)

df['race/ethnicity'].value_counts().plot.pie(autopct="%1.1f%%")



plt.subplot(143)

plt.title('Lunch',fontsize = 20)

df['lunch'].value_counts().plot.pie(autopct="%1.1f%%")



plt.subplot(144)

plt.title('Parentel level of Education',fontsize = 20)

df['parental level of education'].value_counts().plot.pie(autopct="%1.1f%%")

plt.show()
plt.figure(figsize=(15,5))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,

                      wspace=0.5, hspace=0.2)

plt.subplot(131)

plt.title('Math Scores')

sns.barplot(x="gender", y="math score", data=df)

plt.subplot(132)

plt.title('Reading Scores')

sns.barplot(x="gender", y="reading score", data=df)

plt.subplot(133)

plt.title('Writing Scores')

sns.barplot(x="gender", y="writing score", data=df)

plt.show()
plt.figure(figsize=(15,5))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,

                      wspace=0.5, hspace=0.2)

plt.subplot(131)

plt.title('Math Scores')

sns.barplot(hue="gender", y="math score", x="test preparation course", data=df)

plt.subplot(132)

plt.title('Reading Scores')

sns.barplot(hue="gender", y="reading score", x="test preparation course", data=df)

plt.subplot(133)

plt.title('Writing Scores')

sns.barplot(hue="gender", y="writing score", x="test preparation course", data=df)

plt.show()
plt.figure(figsize=(15,5))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,

                      wspace=0.5, hspace=0.2)

plt.subplot(131)

plt.title('Math Scores')

sns.barplot(x="race/ethnicity", y="math score", hue="test preparation course", data=df)

plt.subplot(132)

plt.title('Reading Scores')

sns.barplot(hue="test preparation course", y="reading score", x="race/ethnicity", data=df)

plt.subplot(133)

plt.title('Writing Scores')

sns.barplot(hue="test preparation course", y="writing score", x= 'race/ethnicity',data=df)



plt.show()
plt.figure(figsize=(30,15))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,

                      wspace=0.5, hspace=0.2)

plt.subplot(251)

plt.title('Test Preparation course Vs Gender',fontsize = 15)

sns.countplot(hue="test preparation course", x="gender", data=df)



plt.subplot(254)

plt.title('Test Preparation course Vs Parental Level Of Education',fontsize = 15)

sns.countplot(hue="test preparation course", y="parental level of education", data=df)



plt.subplot(253)

plt.title('Test Preparation course Vs Lunch',fontsize = 15)

sns.countplot(hue="test preparation course", x="lunch", data=df)



plt.subplot(252)

plt.title('Test Preparation course Vs Ethnicity',fontsize = 15)

sns.countplot(hue="test preparation course", y="race/ethnicity", data=df)



plt.show()
plt.title('Gender Vs Ethnicity',fontsize = 20)

sns.countplot(x="gender", hue="race/ethnicity", data=df)

plt.show()
pr=pd.crosstab(df['race/ethnicity'],df['parental level of education'],normalize=0)



pr.plot.bar(stacked=True)

plt.title('Ethinicity Vs Parental Level of Education',fontsize = 20)

plt.show()
plt.figure(figsize=(40,10))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,

                      wspace=0.5, hspace=0.2)

plt.subplot(251)

plt.title('Parental education and Gender',fontsize=15)

sns.countplot(hue="parental level of education", x="gender", data=df)

plt.subplot(252)

plt.title('Parental education and Lunch',fontsize=15)

sns.countplot(hue="parental level of education", x="lunch", data=df)



plt.show()
plt.figure(figsize=(40,10))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,

                      wspace=0.5, hspace=0.2)

plt.subplot(251)

plt.title('Lunch and Gender',fontsize=15)

sns.countplot(x="lunch", hue="gender", data=df)

plt.subplot(252)

plt.title('Ethinicity and Lunch',fontsize=15)

sns.countplot(x="race/ethnicity", hue="lunch", data=df)

plt.show()
df['total marks']=df['math score']+df['reading score']+df['writing score']
df['percentage']=df['total marks']/300*100
#Assigning the grades



def determine_grade(scores):

    if scores >= 85 and scores <= 100:

        return 'Grade A'

    elif scores >= 70 and scores < 85:

        return 'Grade B'

    elif scores >= 55 and scores < 70:

        return 'Grade C'

    elif scores >= 35 and scores < 55:

        return 'Grade D'

    elif scores >= 0 and scores < 35:

        return 'Grade E'

    

df['grades']=df['percentage'].apply(determine_grade)
df.info()
df['grades'].value_counts().plot.pie(autopct="%1.1f%%")

plt.show()
plt.figure(figsize=(30,10))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,

                      wspace=0.5, hspace=0.2)

plt.subplot(251)

plt.title('Grades and Gender')

sns.countplot(hue="gender", x="grades", data=df)



plt.subplot(252)

plt.title('Grades and Lunch')

sns.countplot(hue="lunch", x="grades", data=df)



plt.subplot(253)

plt.title('Grades and Test preperation Course')

sns.countplot(hue="test preparation course", x="grades", data=df)



plt.show()



plt.title('Grades and Parental level of Education',fontsize=20)

sns.countplot(x="parental level of education", hue="grades", data=df)

plt.show()
plt.title('Grades and Ethinicity',fontsize=20)

sns.countplot(x="race/ethnicity", hue="grades", data=df)





gr=pd.crosstab(df['grades'],df['race/ethnicity'],normalize=0) #normalized values 

gr.plot.bar(stacked=True)

plt.title('Grades and Ethinicity',fontsize=20)

plt.show()