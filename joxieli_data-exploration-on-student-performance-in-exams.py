# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

data.head()
data.info()


sns.heatmap(data[['gender','race/ethnicity','parental level of education','lunch','test preparation course','math score','reading score','writing score']].corr(), annot = True)



plt.title('Histogram', fontsize = 30)

plt.show()
sns.set(font_scale=1)

g=sns.countplot(data['parental level of education'])

g.set_xticklabels(g.get_xticklabels(), rotation=30)
sns.countplot(data['race/ethnicity'])
data.describe(include=['O'])
data['average score'] = data[['math score','reading score','writing score']].mean(axis=1)

data.head()
data[['gender', 'average score']].groupby(['gender'], as_index=False).mean().sort_values(by='average score', ascending=False)
data[['race/ethnicity', 'average score']].groupby(['race/ethnicity'], as_index=False).mean().sort_values(by='average score', ascending=False)
data[['parental level of education', 'average score']].groupby(['parental level of education'], as_index=False).mean().sort_values(by='average score', ascending=False)
plt.rcParams['figure.figsize'] = (14, 7)

ax = sns.violinplot(x = data['parental level of education'], y = data['average score'], palette = 'Blues')

ax.set_xlabel(xlabel = 'parental level of education', fontsize = 20)



ax.set_ylabel(ylabel = 'average score', fontsize = 20)

ax.set_title(label = 'Distribution of average score in relation to parental level of education', fontsize = 20)

ax.set_xticklabels(g.get_xticklabels(), rotation=30)

plt.show()
data[['lunch', 'average score']].groupby(['lunch'], as_index=False).mean().sort_values(by='average score', ascending=False)
data.head()
data = pd.concat([data.drop('gender', axis=1), pd.get_dummies(data['gender'])], axis=1)

data = pd.concat([data.drop('race/ethnicity', axis=1), pd.get_dummies(data['race/ethnicity'])], axis=1)

data = pd.concat([data.drop('parental level of education', axis=1), pd.get_dummies(data['parental level of education'])], axis=1)

data = pd.concat([data.drop('lunch', axis=1), pd.get_dummies(data['lunch'])], axis=1)

data = pd.concat([data.drop('test preparation course', axis=1), pd.get_dummies(data['test preparation course'])], axis=1)

data.head()

sns.heatmap(data.corr(),cmap= 'coolwarm')

plt.title('Correlation between variables', fontsize = 30)

plt.show()
data.corr(method ='pearson')