# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

df = pd.read_csv("../input/student-performance/datasets_74977_169835_StudentsPerformance.csv")

df
df.head()
# show the analysis of numerical values.

df.describe()
df.columns
df.isnull().values.any()
df.isnull().sum()
#show count Gender

df['gender'].value_counts()
plt.figure(figsize=(7,7))

sns.barplot(x=df['gender'].value_counts().index,y=df['gender'].value_counts().values)

plt.title('genders other rate')

plt.ylabel('Rates')

plt.legend(loc=0)

plt.show()
plt.figure(figsize=(7,7))

sns.barplot(x=df['race/ethnicity'].value_counts().index,

              y=df['race/ethnicity'].value_counts().values)

plt.xlabel('race/ethnicity')

plt.ylabel('Frequency')

plt.title('Show of Race/Ethnicity Bar Plot')

plt.show()
plt.figure(figsize=(14,7))

sns.barplot(x=df['parental level of education'].value_counts().index,

              y=df['parental level of education'].value_counts().values)

plt.xlabel('parental level of education')

plt.ylabel('Frequency')

plt.title('Show of parental level of education Bar Plot')

plt.show()
import matplotlib.pyplot as plt 

import seaborn as sns  

plt.figure(figsize=(14,6))



plt.subplot(1,3,1)

sns.barplot(x = 'gender', y = 'reading score', data = df)



plt.subplot(1,3,2)

sns.barplot(x = 'gender', y = 'writing score', data = df)



plt.subplot(1,3,3)

sns.barplot(x = 'gender', y = 'math score', data = df)



plt.tight_layout()
sns.lmplot(x='reading score',y='math score',hue='gender',data=df)

plt.xlabel('reading score')  

plt.ylabel('math score')

plt.title('Math Score vs Reading Score')

plt.show()
sns.lmplot(x='writing score',y='math score',hue='gender',data=df)

plt.xlabel('writing score')  

plt.ylabel('math score')

plt.title('Math Score vs Writing Score')

plt.show()
sns.pairplot(df, diag_kind="kde", markers="+",

                  plot_kws=dict(s=50, edgecolor="b", linewidth=1),

                  diag_kws=dict(shade=True))

plt.show()
sns.pairplot(df,kind='reg')

plt.show()
plt.figure(figsize=(16,6))



plt.subplot(1,3,1)

sns.barplot(x = 'race/ethnicity', y = 'reading score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,2)

sns.barplot(x = 'race/ethnicity', y = 'writing score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,3)

sns.barplot(x = 'race/ethnicity', y = 'math score', data = df)

plt.xticks(rotation = 90)



plt.tight_layout()
plt.figure(figsize=(10,7))

sns.barplot(x = "parental level of education", y = "writing score", hue = "gender", data = df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,7))

sns.barplot(x = "parental level of education", y = "reading score", hue = "gender", data = df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,7))

sns.barplot(x = "parental level of education", y = "math score", hue = "gender", data = df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(14,8))

sns.catplot(y="gender", x="math score",

                 hue="parental level of education",

                 data=df, kind="bar")

plt.title('for Parental Level Of Education Gender & Math_Score')

plt.show()
plt.figure(figsize=(16,6))



plt.subplot(1,3,1)

sns.barplot(x = 'test preparation course', y = 'reading score', hue = 'gender', data = df)



plt.subplot(1,3,2)

sns.barplot(x = 'test preparation course', y = 'writing score',hue = 'gender', data = df)



plt.subplot(1,3,3)

sns.barplot(x = 'test preparation course', y = 'math score',hue = 'gender', data = df)



plt.tight_layout()
plt.figure(figsize=(16,6))



plt.subplot(1,3,1)

sns.barplot(x = 'lunch', y = 'reading score', data = df)



plt.subplot(1,3,2)

sns.barplot(x = 'lunch', y = 'writing score', data = df)



plt.subplot(1,3,3)

sns.barplot(x = 'lunch', y = 'math score', data = df)





plt.tight_layout()
# Function to assign grades



def get_grade(marks):

    if marks >= 91:

        return 'O'

    elif marks >= 82 and marks < 91:

        return 'A+'

    elif marks >=73 and marks < 82:

        return 'A'

    elif marks >=64 and marks < 73:

        return 'B+'

    elif marks >= 55 and marks < 64:

        return 'B'

    elif marks >=46 and marks < 55:

        return 'C'

    elif marks >= 35 and marks < 46:

        return 'P'

    elif marks < 35:

        return 'F'
df['reading_grade'] = df['reading score'].apply(get_grade)

df['writing_grade'] = df['writing score'].apply(get_grade)

df['math_grade'] = df['math score'].apply(get_grade)
sns.set_style('whitegrid')

plt.figure(figsize=(16,6))

plt.subplot(1,3,1)

sns.countplot(x ='math_grade', data = df,order = ['O','A+','A','B+','B','C','P','F'],palette='magma')

plt.title('Grade Count in Math')





plt.subplot(1,3,2)

sns.countplot(x ='reading_grade', data = df,order = ['O','A+','A','B+','B','C','P','F'],palette='magma')

plt.title('Grade Count in Reading')



plt.subplot(1,3,3)

sns.countplot(x ='writing_grade', data = df,order = ['O','A+','A','B+','B','C','P','F'],palette='magma')

plt.title('Grade Count in Writing')



plt.tight_layout()
#Number of students having maximum grade in Reading

print('No. of students having maximum grade in reading: ', len(df[df['reading_grade'] == 'O']))

#Number of students having maximum grade in Writing

print('No. of students having maximum grade in writing: ', len(df[df['writing_grade'] == 'O']))

#Number of students having maximum grade in all three categories

perfect_writing = df['writing_grade'] == 'O'

perfect_reading = df['reading_grade'] == 'O'

perfect_math = df['math_grade'] == 'O'



perfect_grade = df[(perfect_math) & (perfect_reading) & (perfect_writing)]

print('Number of students having maximum grade(O) in all three subjects: ',len(perfect_grade))

#Number of students having minimum grade in all three categories

minimum_math = df['math_grade'] == 'F'

minimum_reading = df['reading_grade'] == 'F'

minimum_writing = df['writing_grade'] == 'F'







minimum_grade = df[(minimum_math) & (minimum_reading) & (minimum_writing)]

print('Number of students having minimum grade(F) in all three subjects: ',len(minimum_grade))

#Classifying Students as Passed or Failed

# A student is classified failed if he/she has failed in any one of three subjects otherwise he/she is classified as passed

#Failed Students

failed_students = df[(minimum_math) | (minimum_reading)|(minimum_writing)]

failed = len(failed_students)

print('Total Number of students who failed are: {}' .format(len(failed_students)))
#Passed Students

passed_students = len(df) - len(failed_students)

print('Total Number of students who passed are: {}' .format(passed_students))
f, ax = plt.subplots(figsize=(14, 8))

sns.scatterplot(x="reading score", y="writing score",

                hue="lunch", size="gender",data=df)

plt.show()
f, ax = plt.subplots(figsize=(14, 8))

sns.scatterplot(x="reading score", y="writing score",

                hue="parental level of education", size="gender",data=df)

plt.show()
f, ax = plt.subplots(figsize=(14, 8))

sns.scatterplot(x="reading score", y="writing score",

                hue="test preparation course", size="gender",data=df)

plt.show()
(df[['race/ethnicity','math score', 'reading score', 'writing score']].corr())

ax = sns.heatmap(df.corr(),cmap="Blues",annot=True,annot_kws={"size": 7.5},linewidths=.5)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right");
sns.swarmplot(x=df['lunch'],y=df['reading score'])

plt.show()


sns.swarmplot(x=df['test preparation course'],y=df['math score'],hue=df['gender'])

plt.show()