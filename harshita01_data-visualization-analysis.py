#import all the required libraries. 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

#load the dataset 

df=pd.read_csv('../input/StudentsPerformance.csv')

#check the first 5 values of the data set

df.head()
df.describe()
#check if there is any null value

df.isna().sum()
#visuzlaing the gender

df['gender'].value_counts(normalize=True).plot.bar()

plt.title('Gender Comparison')

plt.xlabel('gender')

plt.ylabel('Counts')

plt.show()
#visualizing the races

df['race/ethnicity'].value_counts(normalize=True).plot.bar()

plt.title('ethnicity comparison')

plt.xlabel('race/ethnicity')

plt.ylabel('Counts')

plt.show()
#visualizing the parental level of education

df['parental level of education'].value_counts(normalize=True).plot.bar()

plt.title('Parenetal education comparison')

plt.xlabel('Degree')

plt.ylabel('counts')

plt.show()
#visualizing the lunch provided

df['lunch'].value_counts(normalize=True).plot.bar()

plt.title('Lunch based comparison')

plt.xlabel('Lunch Types')

plt.ylabel('counts')

plt.show()
#copy the data set before adding more columns

df1= df.copy()
#checking the maximum of each subject

max(df1['math score'])
max(df1['reading score'])
max(df1['writing score'])
#assuming that each subject maximum marks is 100 and calculate the 40% percent of marks as passing_marks

passing_marks= (40/100*100)

passing_marks
#students passed in maths

df1['passed_in_maths']= np.where(df['math score']<passing_marks, 'F', 'P')

df1['passed_in_maths'].value_counts()
#visualization of studetns passed in maths

df1['passed_in_maths'].value_counts(normalize=True).plot.bar()
#students passed in reading 

df1['passed_in_reading']= np.where(df['reading score']<passing_marks, 'F', 'P')

df1['passed_in_reading'].value_counts()
#visualization of students passed in reading

sns.countplot(x=df1['passed_in_reading'], data=df1, palette='bright')
#students passed in writing

df1['passed_in_writing']= np.where(df['writing score']<passing_marks, 'F', 'P')

df1['passed_in_writing'].value_counts()
#visualization of students passed in writing

sns.countplot(x=df1['passed_in_writing'], data=df1, palette='bright')
#Overall passing students in all subjects

df1['overall_pass'] = df1.apply(lambda x : 'P' if x['passed_in_maths'] == 'P' and x['passed_in_reading'] == 'P' and x['passed_in_writing'] == 'P' else 'F', axis =1)

df1['overall_pass'].value_counts()
#check how overall number of passing students depends on gender

Gender=pd.crosstab(df1['gender'], df1['overall_pass'])

Gender
#visualization of gender vs students passed

sns.countplot(x='gender', data=df1, hue='overall_pass', palette='bright')
#check how race affects the overall students passed

race=pd.crosstab(df1['race/ethnicity'], df1['overall_pass'])

race
#visualization of race vs overall students passed

sns.countplot(x='race/ethnicity', data=df1, hue='overall_pass', palette='bright')
#check how parental education controls the passing of students

parental_education=pd.crosstab( df1['parental level of education'], df1['overall_pass'] )

parental_education
#visualization of parental level of education and overall passing students

plt.figure(figsize=(16,6))

sns.countplot(x='parental level of education', data=df1, hue='overall_pass', palette='bright')
#check how lunch controls the passing of students

lunch=pd.crosstab( df1['lunch'], df1['overall_pass'] )

lunch
#visualization of lunch and overall students passed. 

sns.countplot(x='lunch', data=df1, hue='overall_pass', palette='bright')
#check how test preparation course affects the passing of students

test_preparation_course=pd.crosstab( df1['lunch'], df1['overall_pass'] )

test_preparation_course
#visualizaing the relation between test preparation course and overall passed students

sns.countplot(x='test preparation course', data=df1, hue='overall_pass', palette='bright')
#getting total number of marks and plotting on graph

df1['total_marks']= df1['math score']+df1['reading score']+ df1['writing score']

df1['percentage'] = df1['total_marks']/3

plt.figure(figsize=(20,10))

sns.countplot(x='percentage', data=df1)
#function to specify the grade on the basis of percentage obtained.

def Grading(percentage, overall_pass):

        

    if ( percentage >= 80 ):

        return 'A'

    if ( percentage >= 70):

        return 'B'

    if ( percentage >= 60):

        return 'C'

    if ( percentage >= 50):

        return 'D'

    if ( percentage >= 40):

        return 'E'

    if ( overall_pass == 'F'):

        return 'F'

    else: 

        return 'F'



df1['grade'] = df1.apply(lambda x : Grading(x['percentage'], x['overall_pass']), axis=1)
#Check the how many students fall in which grade

df1['grade'].value_counts().sort_index()
#plotting the students performance on pie chart to visualize easily. 

plt.figure(figsize= (20,10))

labels = ['A', 'B', 'C', 'D', 'E', 'F']

numbers= [198, 261, 256, 182, 73, 30]

colors= ['Red', 'Green', 'Blue', 'cyan', 'orange', 'pink']

plt.pie(numbers, labels=labels, colors=colors,startangle=90,autopct='%.1f%%')

plt.legend(labels)

plt.show()
