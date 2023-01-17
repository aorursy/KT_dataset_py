# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
students_performance = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
students_performance.head(10)
students_performance['race/ethnicity'].unique()
students_performance['parental level of education'].unique()
students_performance['lunch'].unique()
students_performance.columns
students_performance.info()
students_performance.describe()
students_performance.boxplot(column='math score',by='race/ethnicity')

plt.show()
students_performance.boxplot(column='reading score',by='race/ethnicity')

plt.show()
students_performance.boxplot(column='writing score',by='race/ethnicity')

plt.show()
students_performance.boxplot(column='math score',by='gender')

plt.show()
students_performance.boxplot(column='reading score',by='gender')

plt.show()
students_performance.boxplot(column='writing score',by='gender')

plt.show()
students_performance.boxplot(column='writing score',by='parental level of education')

plt.xticks(rotation=45)

plt.show()
students_performance.boxplot(column='reading score',by='parental level of education')

plt.xticks(rotation=45)

plt.show()
students_performance.boxplot(column='math score',by='parental level of education')

plt.xticks(rotation=45)

plt.show()
math_score = students_performance['math score']

reading_score = students_performance['reading score']

writing_score = students_performance['writing score']

race_ethnicity = students_performance['race/ethnicity']
data_math = pd.DataFrame({'race/ethnicity': race_ethnicity,'math_score': math_score})

new_index_math = (data_math['math_score'].sort_values(ascending=False)).index.values

sorted_data_math = data_math.reindex(new_index_math)

             
data_reading = pd.DataFrame({'race/ethnicity': race_ethnicity,'reading_score': math_score})

new_index_reading = (data_reading['reading_score'].sort_values(ascending=False)).index.values

sorted_data_reading = data_reading.reindex(new_index_reading)
data_writing = pd.DataFrame({'race/ethnicity': race_ethnicity,'writing_score': math_score})

new_index_writng = (data_writing['writing_score'].sort_values(ascending=False)).index.values

sorted_data_writing = data_writing.reindex(new_index_math)
plt.figure(figsize=(10,10))

sns.barplot(x=students_performance['race/ethnicity'],y= data_writing.writing_score)

plt.xticks(rotation=45)

plt.xlabel('race/ethnicity')

plt.ylabel('writing score')

plt.show()
plt.figure(figsize=(10,10))

sns.barplot(x=students_performance['race/ethnicity'],y= data_reading.reading_score)

plt.xticks(rotation=45)

plt.xlabel('race/ethnicity')

plt.ylabel('reading score')

plt.show()
plt.figure(figsize=(10,10))

sns.barplot(x=students_performance['race/ethnicity'],y= data_math.math_score)

plt.xticks(rotation=45)

plt.xlabel('race/ethnicity')

plt.ylabel('math score')

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(x=students_performance['gender'],y= data_math.math_score)

plt.xticks(rotation=45)

plt.xlabel('race/ethnicity')

plt.ylabel('math score')

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(x=students_performance['gender'],y= data_reading.reading_score)

plt.xticks(rotation=45)

plt.xlabel('race/ethnicity')

plt.ylabel('reading score')

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(x=students_performance['gender'],y= data_writing.writing_score)

plt.xticks(rotation=45)

plt.xlabel('race/ethnicity')

plt.ylabel('writing score')

plt.show()
labels = students_performance['race/ethnicity'].value_counts().index

colors = ['green','red','brown','orange','pink']

explode = [0.1,0.1,0.1,0.1,0.1]

sizes = students_performance['race/ethnicity'].value_counts().values

plt.figure(figsize=(10,10))

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct= '%1f%%')

plt.show()
labels = students_performance['gender'].value_counts().index

colors = ['blue','red']

explode = [0.01,0.01]

sizes = students_performance['gender'].value_counts().values

plt.figure(figsize=(10,10))

plt.pie(sizes,labels=labels,colors=colors,explode=explode,autopct='%1f%%')

plt.show()
students_performance['parental level of education'].unique()
labels = students_performance['parental level of education'].value_counts().index

colors = ['green','red','brown','orange','pink','blue']

explode = [0.1,0.1,0.1,0.1,0.1,0.1]

sizes = students_performance['parental level of education'].value_counts().values

plt.figure(figsize=(10,10))

plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct= '%1f%%')

plt.show()
labels = students_performance['test preparation course'].value_counts().index

colors = ['blue','red']

explode = [0.01,0.01]

sizes = students_performance['test preparation course'].value_counts().values

plt.figure(figsize=(10,10))

plt.pie(sizes,labels=labels,colors=colors,explode=explode,autopct='%1f%%')

plt.show()
labels = students_performance['lunch'].value_counts().index

colors = ['blue','red']

explode = [0.01,0.01]

sizes = students_performance['lunch'].value_counts().values

plt.figure(figsize=(10,10))

plt.pie(sizes,labels=labels,colors=colors,explode=explode,autopct='%1f%%')

plt.show()
sns.pairplot(students_performance)

plt.show()