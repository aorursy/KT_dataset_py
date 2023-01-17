# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.
students = pd.read_csv('../input/StudentsPerformance.csv')
students.head(10)
students.describe()
students.info()
print("---gender---")
print(students.gender.value_counts())
print("--Number of parental level of education--")
print(students['parental level of education'].value_counts())
print("--lunch-type---")
print(students.lunch.value_counts())
print("<<<Test preparation course>>>")
print(students['test preparation course'].value_counts())
print(students['race/ethnicity'].value_counts().plot(kind = 'bar'))
sns.set()
plt.ylabel('Total')
plt.show()
print("-------Female")
print(students['race/ethnicity'][students.gender == 'female'].value_counts(normalize = True))
print("-------Male")
print(students['race/ethnicity'][students.gender == 'male'].value_counts(normalize = True))
print(students.groupby('gender')[['math score','reading score','writing score']].agg(['max','min']))
print(students.groupby('gender')['math score','reading score','writing score'].mean())
female = students[(students.gender == 'female') & (students['race/ethnicity'] == 'group C')]
male = students[(students.gender == 'male') & (students['race/ethnicity'] == 'group C')]
print("----females----")
print(female['parental level of education'].value_counts(normalize = True))
print("----Males---")
print(male['parental level of education'].value_counts(normalize = True))
female1 = students[students.gender == 'female']
male1 = students[students.gender == 'male']

print("--female--")
print(female1['race/ethnicity'].value_counts())
print("--male--")
print(male1['race/ethnicity'].value_counts())
print("---female--")
print(female1['lunch'].value_counts(normalize = True))
print("---male--")
print(male1.lunch.value_counts(normalize = True))
print("---female--")
print(female1['test preparation course'].value_counts(normalize = True))
print("---male--")
print(male1['test preparation course'].value_counts(normalize = True))
print("--female--")
print(female1.groupby(['lunch','test preparation course'])['math score'].max().unstack(level = 'lunch'))
print("--male--")
print(male1.groupby(['lunch','test preparation course'])['math score'].max().unstack(level = 'lunch'))
print(female1.groupby('race/ethnicity')['math score'].agg(['min','max']))
print(male1.groupby('race/ethnicity')['math score'].agg(['min','max']))
print(female1.groupby('lunch')['math score'].agg(['min','max']))
print(male1.groupby('lunch')['math score'].agg(['min','max']))
print("---Female---")
groupE = students[(students.gender == 'female') & (students['race/ethnicity'] == 'group E')]
groupC = students[(students.gender == 'female') & (students['race/ethnicity'] == 'group C')]
print(groupE.lunch.value_counts(normalize=True))
print(groupC.lunch.value_counts(normalize=True))
print('---courses---')
print(groupE['test preparation course'].value_counts(normalize=True))
print(groupC['test preparation course'].value_counts(normalize=True))
print("----Male---")
groupE = students[(students.gender == 'male') & (students['race/ethnicity'] == 'group E')]
groupC = students[(students.gender == 'male') & (students['race/ethnicity'] == 'group C')]
print(groupE.lunch.value_counts(normalize=True))
print(groupC.lunch.value_counts(normalize=True))
print("---Courses on groupE---")
print(groupE['test preparation course'].value_counts(normalize=True))
print("---Courses on groupC---")
print(groupC['test preparation course'].value_counts(normalize=True))
students.head()
plt.subplot(1,2,1)
sns.violinplot(y = 'writing score', data = students)
plt.subplot(1,2,2)
sns.violinplot(y = 'reading score' ,data = students)
plt.tight_layout()
plt.show()
print(students.groupby('gender')['reading score','writing score'].agg(['min','max']))
reading_score = students[(students['reading score'] == 100)]
writing_score = students[(students['writing score'] == 100)]
print(reading_score.groupby('gender')['reading score'].value_counts().plot(kind = 'bar'))
plt.show()
print(writing_score.groupby('gender')['writing score'].value_counts().plot(kind = 'bar'))
plt.show()
sns.countplot(x = 'race/ethnicity' , data = reading_score,hue = 'gender',palette='dark')
plt.show()
sns.countplot(x = 'race/ethnicity' , data = writing_score , hue = 'gender' , palette= 'dark')
plt.show()
male1.groupby(['lunch','test preparation course'])['writing score','reading score','math score'].mean()
female1.groupby(['lunch','test preparation course'])['writing score','reading score','math score'].mean()
