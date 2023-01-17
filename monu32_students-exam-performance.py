# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
df=pd.read_csv('../input/StudentsPerformance.csv')
df
df.columns=['gender','race','level_of_education','lunch','preparation','math','reading','writing']

df['percentage']=(df['math']+df['reading']+df['writing'])/3
df
# Check information of dataset
df.info()
# Dataset doesn't contain null values 
from matplotlib import pyplot as plt

#Check frequency of students with respect to gender

plt.bar('male',df.gender[df.gender=='male'].count())
plt.bar('female',df.gender[df.gender=='female'].count())
plt.ylabel("Number of Students")
plt.show()
def For_assign_grade(percentage):      # Define funtion for assign grade of students
    if percentage>=90:
        return 'A'
    elif percentage>=80:
        return 'B'
    elif percentage>=70:
        return 'C'
    elif percentage>=60:
        return 'D'
    elif percentage>=50:
        return 'E'
    elif percentage>=40:
        return 'F'
    else:
        return 'FAIL'

df['grade']=df.percentage.apply(lambda x:For_assign_grade(x))
df
plt.bar('Pass',df.grade[df.grade!='FAIL'].count())
plt.bar('Fail',df.grade[df.grade=='FAIL'].count())
plt.ylabel('Number of Students')
plt.show()
# Number of Students who were fail in the exam are very less as compare to students who pass the exam
plt.bar('Grade A',df.grade[df.grade=='A'].count())
plt.bar('Grade B',df.grade[df.grade=='B'].count())
plt.bar('Grade C',df.grade[df.grade=='C'].count())
plt.bar('Grade D',df.grade[df.grade=='D'].count())
plt.bar('Grade E',df.grade[df.grade=='E'].count())
plt.bar('Grade F',df.grade[df.grade=='F'].count())
plt.bar('Fail',df.grade[df.grade=='FAIL'].count())
plt.legend(['90-100','80-89','70-79','60-69','50-59','40-49','Fail'])
plt.show()
#We can see Grade C have the highest peak and Grade D have approx equal to Grade C
import seaborn as sns
p=sns.countplot(x='grade',data=df,hue='gender')
plt.xlabel('Grade')
# In top three grades female dominate male in result
p=sns.countplot(x='race',data=df,palette='bright')
_=plt.xlabel('Race / Ethinicity')
# Group C contain most number of students 
# All unique valeus in race column
df.race.unique()
p=sns.countplot(x='race',data=df,hue='grade',palette='bright')
plt.xlabel('\nRace / Ethinicity')
# Here Grade C and Grade D have higher peaks in every group except group A
# Group B contain more proportion of fail in exam as compare to other groups
p=sns.countplot(x='level_of_education',data=df,hue='grade')
unused_variable=plt.setp(p.get_xticklabels(),rotation=330)
unused_variable=plt.xlabel('\nParental level of Education')
# All level of education contain approx equal peaks of grade A and grade D
# Failure of student happen more,when parental education is either high school or some high school
# Failure of student is NIL,if parental education is master's degree
# We see that race/ethinicty and parental level education affect differently on grade
p=sns.countplot(x='race',data=df,hue='level_of_education',palette='bright')
unused_variale=plt.xlabel('Race / Ethinicity')
unused_variale=plt.legend(loc='upper right')
# In above graph ,high school and some high school contain nearly equal number of frequency.
p=sns.countplot(x='grade',data=df,hue='preparation')
unused_variale=plt.xlabel('Parental level of Education')
unused_variale=plt.legend(loc='upper right')
a=sns.scatterplot(x="math",y='percentage',data=df)
_=plt.xlabel('Math')
_=plt.ylabel('Percentage')
a=sns.scatterplot(x="reading",y='percentage',data=df)
_=plt.xlabel('Reading')
_=plt.ylabel('Percentage')
a=sns.scatterplot(x="writing",y='percentage',data=df)
_=plt.xlabel('Writing')
_=plt.ylabel('Percentage')
# Graph of math and percentage contain more outlier as compare to other scatter plots

