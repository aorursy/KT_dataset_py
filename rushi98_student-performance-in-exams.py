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
df=pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
df.isnull().sum()
df['total score'] = df.loc[:,['math score', 'reading score', 'writing score']].sum(axis=1)
df.head()
print(' Number of records: {} \n Number of columns: {}'.format(df.shape[0], df.shape[1]))
import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns
print('The highest score in maths is {}'.format(df['math score'].max()))

print('The highest score in reading is {}'.format(df['reading score'].max()))

print('The highest score in writing is {}'.format(df['writing score'].max()))



print('The highest total score is {}'.format(df['total score'].max()))
print('The average score in maths is {}'.format(np.mean(df['math score'])))

print('The average score in reading is {}'.format(np.mean(df['reading score'])))

print('The average score in writing is {}'.format(np.mean(df['writing score'])))



print('The average score in total is {}'.format(np.mean(df['total score'])))
m_a = df.loc[df['math score'] >= 50, 'math score'] 

m_b = df.loc[df['math score'] < 50, 'math score']

len_m_a = len(m_a)

len_m_b = len(m_b)

my_labels = 'marks >= 50','marks < 50'

plt.pie([len_m_a, len_m_b], labels=my_labels, autopct='%1.1f%%')

plt.title('Students performance in Math')

plt.axis('equal')

plt.show()



r_a = df.loc[df['reading score'] >= 50, 'reading score'] 

r_b = df.loc[df['reading score'] < 50, 'reading score']

len_r_a = len(r_a)

len_r_b = len(r_b)

my_labels = 'marks >= 50','marks < 50'

my_colors = ['lightblue', 'lightsteelblue']

plt.pie([len_r_a, len_r_b], labels=my_labels, autopct='%1.1f%%', colors=my_colors)

plt.title('Students performance in reading')

plt.axis('equal')

plt.show()





w_a = df.loc[df['writing score'] >= 50, 'writing score'] 

w_b = df.loc[df['writing score'] < 50, 'writing score']

len_w_a = len(w_a)

len_w_b = len(w_b)

my_labels = 'marks >= 50','marks < 50'

my_colors = ['yellow', 'red']

plt.pie([len_w_a, len_w_b], labels=my_labels, autopct='%1.1f%%')

plt.title('Students performance in writing')

plt.axis('equal')

plt.show()
ax = sns.countplot(x='gender', data = df, hue = 'test preparation course', palette="Set3")
df['Top scorer'] = np.where((df['math score'] >= 85) & (df['reading score'] >= 85) & (df['writing score'] >= 85),'yes','no')
df.head()
ax_1 = sns.countplot(x='gender', data = df, hue = 'Top scorer', palette="Set3")
df['Result'] = np.where((df['math score'] >= 40) & (df['reading score'] >=40) & (df['writing score'] >=40),'pass','fail')
df.head()
ax_1 = sns.countplot(x='parental level of education', data = df, hue = 'Result')

ax_1.set_xticklabels(ax_1.get_xticklabels(), rotation=40, ha="right")
ax_2 = sns.countplot(x='parental level of education', data = df, hue = 'Top scorer')

ax_2.set_xticklabels(ax_2.get_xticklabels(), rotation=40, ha="right")
df['avg_score'] = df.loc[:,['math score', 'reading score', 'writing score']].mean(axis=1)
df.head()
df['avg_score'] = df['avg_score'].round(decimals=2)
df.head()
print('Average Score of Male students: {}'.format(df.loc[df['gender']=='male', 'avg_score'].mean()))

print('Average Score of Female students: {}'.format(df.loc[df['gender']=='female', 'avg_score'].mean()))
df[['gender', 'race/ethnicity', 'parental level of education','avg_score']].sort_values(by='avg_score',ascending=False).head(10).style.background_gradient("YlGn")
df[['gender', 'race/ethnicity', 'parental level of education','avg_score']].sort_values(by='avg_score',ascending=True).head(10).style.background_gradient("Reds")
correlation=df[["math score","reading score","writing score","avg_score"]].corr()

sns.heatmap(correlation,mask=np.triu(np.ones_like(correlation,dtype=np.bool)),cmap="RdBu_r",annot=True)

plt.show()