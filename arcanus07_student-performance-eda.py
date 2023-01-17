# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
df['final score'] = df['math score'] + df['reading score'] + df['writing score']
df.drop('lunch', axis = 1, inplace = True)
df
df.isnull().sum()
df.describe()
math_pass = df['math score'].quantile(0.95)

math_pass
df['Math_PassStatus'] = np.where(df['math score']< math_pass, 'F', 'P')

df.head()
reading_pass = df['reading score'].quantile(0.90)

reading_pass
df['Reading_PassStatus'] = np.where(df['reading score']< reading_pass, 'F', 'P')

df.head()
writing_pass = df['writing score'].quantile(0.95)

writing_pass
df['Writing_PassStatus'] = np.where(df['writing score']< writing_pass, 'F', 'P')

df.head()
final_pass = df['final score'].quantile(0.9)

final_pass
df['Final_PassStatus'] = np.where((df['final score'] > writing_pass) & (df['Math_PassStatus'] == 'P') & (df['Reading_PassStatus'] == 'P') & (df['Writing_PassStatus'] == 'P'), 'P', 'F')

df.head()
df['Final_PassStatus'][df['Final_PassStatus'] == 'P'].value_counts()
passed = df[df['Final_PassStatus'] == 'P'].copy()

passed
failed = df[df['Final_PassStatus'] == 'F'].copy()

failed
passed
import matplotlib.pyplot as plt

import seaborn as sns

plt.title('Gender Distribution')

sns.countplot(x="gender", data = df, palette="bright")

plt.show()
plt.title('Race Distribution')

sns.countplot(x="race/ethnicity", data = df, palette="bright")

plt.show()
plt.figure(figsize = (10,10))

plt.title('Parental Education Distribution')

sns.countplot(x="parental level of education", data = df, palette="bright")

plt.show()
plt.title('Test Preparation Distribution')

sns.countplot(x="test preparation course", data = df, palette="bright")

plt.show()
plt.figure(figsize = (10,10))

plt.title('Gender Distribution')

sns.countplot(x="gender", data = passed, palette="bright")

plt.show()
f, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True)

plt.title('Race Distribution')

axes[0].set_title('Distribution of Race in Passed Students')

sns.countplot(x="race/ethnicity", data = passed, palette="bright", ax = axes[0])

axes[1].set_title('Distribution of Race in Failed Students')

sns.countplot(x="race/ethnicity", data = failed, palette="bright", ax = axes[1])

plt.show()
plt.title('Distribution of Race in Passed Students')

sns.countplot(x="race/ethnicity", data = passed, palette="bright")

plt.show()
plt.title('Distribution of Race in Failed Students')

sns.countplot(x="race/ethnicity", data = failed, palette="bright")

plt.show()
plt.figure(figsize=(20,10))

plt.title("Distribution of Students' Results by Parental Education")

sns.countplot(x='parental level of education', data = df, hue='Final_PassStatus', palette='bright')

plt.plot()
plt.figure(figsize=(20,10))

plt.title("Distribution of Students' Results by Race")

sns.countplot(x='race/ethnicity', data = df, hue='Final_PassStatus', palette='bright')

plt.plot()
plt.figure(figsize=(20,10))

plt.title("Distribution of Students' Results by Test Preparation ")

sns.countplot(x='test preparation course', data = df, hue='Final_PassStatus', palette='bright')

plt.plot()
plt.figure(figsize = (20,10))

df['Final_PassStatus'].value_counts().plot.pie(colors = ['lightblue', 'lightgreen'],autopct='%1.0f%%')

plt.plot()
plt.figure(figsize = (10,10))

sns.distplot(df['final score'], color = 'magenta')

plt.title('Comparison of Total Score of All Students', fontweight = 30, fontsize = 20)

plt.xlabel('total score')

plt.ylabel('count')

plt.show()
plt.figure(figsize = (10,10))

sns.distplot(df['math score'], color = 'magenta')

plt.title('Comparison of Math Score of All Students', fontweight = 30, fontsize = 20)

plt.xlabel('math score')

plt.ylabel('count')

plt.show()
plt.figure(figsize = (10,10))

sns.distplot(df['reading score'], color = 'magenta')

plt.title('Comparison of Reading Score of All Students', fontweight = 30, fontsize = 20)

plt.xlabel('reading score')

plt.ylabel('count')

plt.show()
plt.figure(figsize = (10,10))

sns.distplot(df['writing score'], color = 'magenta')

plt.title('Comparison of Writing Score of All Students', fontweight = 30, fontsize = 20)

plt.xlabel('writing score')

plt.ylabel('count')

plt.show()
plt.figure(figsize=(20,10))

sns.countplot(x='parental level of education', data = df, hue='race/ethnicity', order = ['some high school', 'high school', 'some college', 

                "associate's degree", "bachelor's degree", "master's degree"], palette='bright')

plt.title('Relation Between Race and Parental Education')

plt.plot()
plt.figure(figsize=(20,10))

sns.countplot(x='test preparation course', data = df, hue='race/ethnicity', palette='bright')

plt.title('Relation Between Race and Test Preparation')

plt.plot()
fig = plt.figure(figsize = (20,20))

# plt.subplot(3,2,1)

ax1 = fig.add_subplot(3,2,1)

ax1.title.set_text('Group A')

df['parental level of education'][df['race/ethnicity'] == 'group A'].value_counts().plot.pie(colors = ['lightblue', 'lightgreen'],autopct='%1.0f%%')

# plt.subplot(3,2,2)

ax1 = fig.add_subplot(3,2,2)

ax1.title.set_text('Group B')

df['parental level of education'][df['race/ethnicity'] == 'group B'].value_counts().plot.pie(colors = ['lightblue', 'lightgreen'],autopct='%1.0f%%')

# plt.subplot(3,2,3)

ax1 = fig.add_subplot(3,2,3)

ax1.title.set_text('Group C')

df['parental level of education'][df['race/ethnicity'] == 'group C'].value_counts().plot.pie(colors = ['lightblue', 'lightgreen'],autopct='%1.0f%%')

# plt.subplot(3,2,4)

ax1 = fig.add_subplot(3,2,4)

ax1.title.set_text('Group D')

df['parental level of education'][df['race/ethnicity'] == 'group D'].value_counts().plot.pie(colors = ['lightblue', 'lightgreen'],autopct='%1.0f%%')

# plt.subplot(3,2,5)

ax1 = fig.add_subplot(3,2,5)

ax1.title.set_text('Group E')

df['parental level of education'][df['race/ethnicity'] == 'group E'].value_counts().plot.pie(colors = ['lightblue', 'lightgreen'],autopct='%1.0f%%')

plt.plot()
fig = plt.figure(figsize = (20,20))

# plt.subplot(3,2,1)

plt.title('Race')

ax1 = fig.add_subplot(1,2,1)

ax1.title.set_text('Pass')

df['race/ethnicity'][df['Final_PassStatus'] == 'P'].value_counts().plot.pie(colors = ['lightblue', 'lightgreen'],autopct='%1.0f%%')

ax1 = fig.add_subplot(1,2,2)

ax1.title.set_text('Fail')

df['race/ethnicity'][df['Final_PassStatus'] == 'F'].value_counts().plot.pie(colors = ['lightblue', 'lightgreen'],autopct='%1.0f%%')

plt.plot()
plt.figure(figsize = (10,20))

failed['test preparation course'].value_counts().plot.pie(colors = ['lightblue', 'lightgreen'],autopct='%1.0f%%')

plt.plot()
plt.figure(figsize = (10,20))

passed['test preparation course'].value_counts().plot.pie(colors = ['lightblue', 'lightgreen'],autopct='%1.0f%%')

plt.plot()
plt.figure(figsize = (10,20))

failed['race/ethnicity'][failed['test preparation course'] == 'none'].value_counts().plot.pie(colors = ['lightblue', 'lightgreen'],autopct='%1.0f%%')

plt.plot()