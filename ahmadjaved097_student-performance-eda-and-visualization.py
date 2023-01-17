import warnings                       # to hide warnings if any

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))

import pandas as pd                #Data Processing

import numpy as np                 # Linear Algebra

import matplotlib.pyplot as plt    # Data Visualization

import seaborn as sns              # Data Visualization



%matplotlib inline
df = pd.read_csv('../input/StudentsPerformance.csv')

df.head(3)
df.info()
df.isnull().sum()
df.shape
df.describe()
count  = 0

for i in df['gender'].unique():

    count = count + 1

    print(count,'. ',i)
count = 0

for i in sorted(df['race/ethnicity'].unique()):

    count = count + 1

    print(count, '. ',i)

print('Number of different races/ethnicity of people: ', df['race/ethnicity'].nunique())
count = 0

for i in df['parental level of education'].unique():

    count = count + 1

    print(count, '. ', i)
count  = 0

for i in df['lunch'].unique():

    count = count + 1

    print(count,'. ',i)
count  = 0

for i in df['test preparation course'].unique():

    count = count + 1

    print(count,'.',i)
sns.set_style('darkgrid')
sns.pairplot(df, hue = 'gender')

plt.show()
sns.heatmap(df.corr(), annot = True, cmap='inferno')

plt.show()
plt.figure(figsize=(8,5))

sns.distplot(df['math score'], kde = False, color='m', bins = 30)

plt.ylabel('Frequency')

plt.title('Math Score Distribution')

plt.show()
plt.figure(figsize=(8,5))

sns.distplot(df['reading score'], kde = False, color='r', bins = 30)

plt.ylabel('Frequency')

plt.title('Reading Score Distribution')

plt.show()
plt.figure(figsize=(8,5))

sns.distplot(df['writing score'], kde = False, color='blue', bins = 30)

plt.ylabel('Frequency')

plt.title('Writing Score Distribution')

plt.show()
print('Maximum score in Maths is: ',max(df['math score']))

print('Minimum score in Maths is: ',min(df['math score']))
print('Maximum score in Reading is: ',max(df['reading score']))

print('Minimum score in Reading is: ',min(df['reading score']))
print('Maximum score in Writing is: ',max(df['writing score']))

print('Mimimum score in Writing is: ',min(df['writing score']))
print('No. of students having maximum score in math: ', len(df[df['math score'] == 100]))
print('No. of students having maximum score in reading: ', len(df[df['reading score'] == 100]))
print('No. of students having maximum score in writing: ', len(df[df['writing score'] == 100]))
perfect_writing = df['writing score'] == 100

perfect_reading = df['reading score'] == 100

perfect_math = df['math score'] == 100



perfect_score = df[(perfect_math) & (perfect_reading) & (perfect_writing)]

perfect_score
print('Number of students having maximum marks in all three subjects: ',len(perfect_score))
minimum_math = df['math score'] == 0

minimum_reading = df['reading score'] == 17

minimum_writing = df['writing score'] == 10







minimum_score = df[(minimum_math) & (minimum_reading) & (minimum_writing)]

minimum_score
print('No. of students having minimum marks in all three subjects: ', len(minimum_score))
plt.figure(figsize=(10,4))



plt.subplot(1,3,1)

sns.barplot(x = 'gender', y = 'reading score', data = df)



plt.subplot(1,3,2)

sns.barplot(x = 'gender', y = 'writing score', data = df)



plt.subplot(1,3,3)

sns.barplot(x = 'gender', y = 'math score', data = df)



plt.tight_layout()
plt.figure(figsize=(14,4))



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

plt.figure(figsize=(14,4))



plt.subplot(1,3,1)

sns.barplot(x = 'test preparation course', y = 'reading score', hue = 'gender', data = df)



plt.subplot(1,3,2)

sns.barplot(x = 'test preparation course', y = 'writing score',hue = 'gender', data = df)



plt.subplot(1,3,3)

sns.barplot(x = 'test preparation course', y = 'math score',hue = 'gender', data = df)



plt.tight_layout()
plt.figure(figsize=(13,5))



plt.subplot(1,3,1)

sns.barplot(x = 'parental level of education', y = 'reading score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,2)

sns.barplot(x = 'parental level of education', y = 'writing score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,3)

sns.barplot(x = 'parental level of education', y = 'math score', data = df)

plt.xticks(rotation = 90)



plt.tight_layout()
plt.figure(figsize=(14,4))



plt.subplot(1,3,1)

sns.barplot(x = 'lunch', y = 'reading score', data = df)



plt.subplot(1,3,2)

sns.barplot(x = 'lunch', y = 'writing score', data = df)



plt.subplot(1,3,3)

sns.barplot(x = 'lunch', y = 'math score', data = df)





plt.tight_layout()
print('----Females----')

print('Max. math Score: ', df[df['gender'] == 'female']['math score'].max())

print('Min. math Score: ', df[df['gender'] == 'female']['math score'].min())

print('Average math Score: ', df[df['gender'] == 'female']['math score'].mean())

print('----Males----')

print('Max. math Score: ', df[df['gender'] == 'male']['math score'].max())

print('Min. math Score: ', df[df['gender'] == 'male']['math score'].min())

print('Average math Score: ', df[df['gender'] == 'male']['math score'].mean())
print('----Females----')

print('Max. reading Score: ', df[df['gender'] == 'female']['reading score'].max())

print('Min. reading Score: ', df[df['gender'] == 'female']['reading score'].min())

print('Average reading Score: ', df[df['gender'] == 'female']['reading score'].mean())

print('----Males----')

print('Max. reading Score: ', df[df['gender'] == 'male']['reading score'].max())

print('Min. reading Score: ', df[df['gender'] == 'male']['reading score'].min())

print('Average reading Score: ', df[df['gender'] == 'male']['reading score'].mean())
print('----Females----')

print('Max. writing Score: ', df[df['gender'] == 'female']['writing score'].max())

print('Min. writing Score: ', df[df['gender'] == 'female']['writing score'].min())

print('Average writing Score: ', df[df['gender'] == 'female']['writing score'].mean())

print('----Males----')

print('Max. writing Score: ', df[df['gender'] == 'male']['writing score'].max())

print('Min. writing Score: ', df[df['gender'] == 'male']['writing score'].min())

print('Average writing Score: ', df[df['gender'] == 'male']['writing score'].mean())
plt.figure(figsize=(12,5))



plt.subplot(1,3,1)

sns.boxplot(x = 'gender', y = 'math score', data = df,palette = ['coral', 'lawngreen'])



plt.subplot(1,3,2)

sns.boxplot(x = 'gender', y = 'reading score', data = df,palette = ['coral', 'lawngreen'])



plt.subplot(1,3,3)

sns.boxplot(x = 'gender', y = 'writing score', data = df,palette = ['coral', 'lawngreen'])



plt.tight_layout()
for i in sorted(df['race/ethnicity'].unique()):

    print('-----',i,'-----')

    print('Max. marks: ', df[df['race/ethnicity'] == i]['math score'].max())

    print('Min. marks: ', df[df['race/ethnicity'] == i]['math score'].min())

    print('Average marks: ', df[df['race/ethnicity'] == i]['math score'].mean())
for i in sorted(df['race/ethnicity'].unique()):

    print('-----',i,'-----')

    print('Max. marks: ', df[df['race/ethnicity'] == i]['reading score'].max())

    print('Min. marks: ', df[df['race/ethnicity'] == i]['reading score'].min())

    print('Average marks: ', df[df['race/ethnicity'] == i]['reading score'].mean())
for i in sorted(df['race/ethnicity'].unique()):

    print('-----',i,'-----')

    print('Max. marks: ', df[df['race/ethnicity'] == i]['writing score'].max())

    print('Min. marks: ', df[df['race/ethnicity'] == i]['writing score'].min())

    print('Average marks: ', df[df['race/ethnicity'] == i]['writing score'].mean())
plt.figure(figsize=(14,5))

plt.subplot(1,3,1)

sns.boxplot(x = 'race/ethnicity', y = 'math score', data = df)



plt.subplot(1,3,2)

sns.boxplot(x = 'race/ethnicity', y = 'reading score', data = df)



plt.subplot(1,3,3)

sns.boxplot(x = 'race/ethnicity', y = 'writing score', data = df)



plt.tight_layout()
for i in df['parental level of education'].unique():

    print('-----',i,'-----')

    print('Max. marks: ', df[df['parental level of education'] == i]['math score'].max())

    print('Min. marks: ', df[df['parental level of education'] == i]['math score'].min())

    print('Average. marks: ', df[df['parental level of education'] == i]['math score'].mean())

    
for i in df['parental level of education'].unique():

    print('-----',i,'-----')

    print('Max. marks: ', df[df['parental level of education'] == i]['reading score'].max())

    print('Min. marks: ', df[df['parental level of education'] == i]['reading score'].min())

    print('Average. marks: ', df[df['parental level of education'] == i]['reading score'].mean())

    
for i in df['parental level of education'].unique():

    print('-----',i,'-----')

    print('Max. marks: ', df[df['parental level of education'] == i]['writing score'].max())

    print('Min. marks: ', df[df['parental level of education'] == i]['writing score'].min())

    print('Average. marks: ', df[df['parental level of education'] == i]['writing score'].mean())

    
sns.set_style('whitegrid')

plt.figure(figsize=(16,7))

plt.subplot(1,3,1)

sns.boxplot(x ='parental level of education' , y = 'math score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,2)

sns.boxplot(x ='parental level of education' , y = 'reading score', data = df)

plt.xticks(rotation = 90)



plt.subplot(1,3,3)

sns.boxplot(x ='parental level of education' , y = 'writing score', data = df)

plt.xticks(rotation = 90)



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

plt.figure(figsize=(16,5))

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
print('-------- GRADE STATISTICS --------')

print('==== MATH GRADE ====')

print(df['math_grade'].value_counts())

print('==== READING GRADE ====')

print(df['reading_grade'].value_counts())

print('==== WRITING GRADE ====')

print(df['writing_grade'].value_counts())
print('No. of students having maximum grade in math: ', len(df[df['math_grade'] == 'O']))
print('No. of students having maximum grade in reading: ', len(df[df['reading_grade'] == 'O']))
print('No. of students having maximum grade in writing: ', len(df[df['writing_grade'] == 'O']))
perfect_writing = df['writing_grade'] == 'O'

perfect_reading = df['reading_grade'] == 'O'

perfect_math = df['math_grade'] == 'O'



perfect_grade = df[(perfect_math) & (perfect_reading) & (perfect_writing)]

print('Number of students having maximum grade(O) in all three subjects: ',len(perfect_grade))
minimum_math = df['math_grade'] == 'F'

minimum_reading = df['reading_grade'] == 'F'

minimum_writing = df['writing_grade'] == 'F'







minimum_grade = df[(minimum_math) & (minimum_reading) & (minimum_writing)]

print('Number of students having minimum grade(F) in all three subjects: ',len(minimum_grade))
#Failed Students

failed_students = df[(minimum_math) | (minimum_reading)|(minimum_writing)]

failed = len(failed_students)

print('Total Number of students who failed are: {}' .format(len(failed_students)))
#Passed Students

passed_students = len(df) - len(failed_students)

print('Total Number of students who passed are: {}' .format(passed_students))
plt.figure(figsize=(8,6))



#Data to plot

labels = 'Passed Students', 'Failed Students'

sizes = [passed_students,failed]

colors = ['skyblue','yellowgreen']

explode = (.2,0)



#Plot

plt.pie(sizes,explode = explode, labels = labels,colors = colors,

       autopct='%1.1f%%',shadow = True, startangle=360)

plt.axis('equal')

plt.title('Percentage of Students who passed/failed in Exams')

plt.show()