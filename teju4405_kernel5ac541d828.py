# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
performance=pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

performance.head()
performance.describe()
performance.info()
student=performance.drop(['test preparation course','lunch','race/ethnicity'],axis=1)

student
student.iloc[5:995]
student.plot( figsize=(5,5),subplots=True)
lm=LinearRegression()

x=student['math score'].values.reshape(-1,1)

y=student['writing score'].values.reshape(-1,1)

f=lm.fit(x,y)

z=lm.predict(x)

z

lm.coef_
lm.intercept_
plt.scatter(x,y)
student.corr()
math_score=(student['math score']-student['math score'].mean())/student['math score'].std()

reading_score=(student['reading score']-student['reading score'].mean())/student['reading score'].std()

writing_score=(student['writing score']-student['writing score'].mean())/student['writing score'].std()

math_score
reading_score
writing_score
male_candidate=student['gender']=='male'

male_candidate.value_counts()
Pass_maths=student['math score']>=50

Pass_maths.value_counts()

Pass_maths.mean()
Pass_reading=student['reading score']>=50

Pass_reading.value_counts()
Pass_reading.mean()
Pass_writing=student['writing score']>=50

Pass_writing.value_counts()
Pass_writing.mean()
student.plot.hist(figsize=(17,7),alpha=0.6)

plt.xlabel('Marks obtained')

plt.ylabel('Number of students')

plt.show()
student.plot.kde(figsize=(6,6))
student.plot.scatter(x='math score',y='reading score')

student.plot.scatter(x='math score',y='writing score')
student.plot.scatter(x='writing score',y='reading score')
group=student.groupby('gender')['math score','reading score','writing score'].mean()

group
group.plot()
group.plot.pie(figsize=(12,10),subplots=True,autopct='%1.1f%%',)