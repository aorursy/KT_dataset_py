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
df=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

df

df.columns=df.columns.str.strip().str.lower().str.replace(' ','_')

df
df.info()
from collections import Counter

ethnicity=Counter(df['race/ethnicity'])

ethnicity_most_common = ethnicity.most_common(5)

ethnicity_most_common

x,y= zip(*ethnicity_most_common)

x,y= list(x), list(y)

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(15,10))

ax=sns.barplot(x=x,y=y,palette=sns.hls_palette(len(x)))

plt.xlabel('Ethnic Groups')

plt.ylabel('How many people are there')
df.rename(columns={'race/ethnicity':'ethnicity'}, inplace=True)

df.columns
labels=df.ethnicity.value_counts()

labels

colors=['grey','blue','red','yellow','green']

explode=[0,0,0,0,0]

plt.figure(figsize=(8,8))

plt.pie(labels, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%')

test_prep=df['test_preparation_course'].value_counts()

test_prep
plt.figure(figsize=(10,7))

sns.barplot(x='parental_level_of_education',y='math_score', hue='gender', data=df)

plt.show()
plt.figure(figsize=(10,7))

sns.barplot(x='gender',y='math_score', hue='ethnicity', data=df)

plt.show()
pal=sns.cubehelix_palette(2,rot=-.5,dark=.3)

sns.violinplot(data=df, palette=pal, inner='points')

plt.show()
plt.subplots(figsize=(8,5))

sns.swarmplot(x='test_preparation_course',y='math_score',hue='ethnicity',data=df)

plt.show()
f,ax=plt.subplots(figsize=(7,7))

sns.heatmap(df.corr(),annot=True, linewidth=.5, fmt='.1f', ax=ax)

plt.show()
sns.kdeplot(df.writing_score, df.reading_score , cut=3)

plt.show()
sns.pairplot(df)

plt.show()