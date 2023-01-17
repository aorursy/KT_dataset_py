# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
train.head()
import missingno as msno
plt.figure(figsize=(5,5))
msno.matrix(train)
plt.savefig('f18')
train.dtypes

train.describe()
train.info()
train.isnull().sum()
train.gender.value_counts()
plt.title('female vs male math score distribution')
sns.distplot(train[train.gender=='female']['math score'],label='female')
sns.distplot(train[train.gender=='male']['math score'],label='male')
plt.legend(['female','male'])
plt.savefig('f1')
plt.title('female vs male reading score distribution')
sns.distplot(train[train.gender=='female']['reading score'],label='female')
sns.distplot(train[train.gender=='male']['reading score'],label='male')
plt.legend()
plt.savefig('f1')
plt.title('female vs male writing score distribution')
sns.distplot(train[train.gender=='female']['writing score'],label='female')
sns.distplot(train[train.gender=='male']['writing score'],label='male')
plt.legend()
plt.savefig('f2')

sns.scatterplot(x='math score',y='reading score',hue='gender',data=train)
plt.savefig('f4')
sns.scatterplot(x='writing score',y='reading score',hue='gender',data=train)
plt.savefig('f5')
sns.scatterplot(x='math score',y='reading score',hue='gender',data=train)
plt.savefig('f6')
sns.pairplot(train,hue='gender')
plt.savefig('f7')
train['race/ethnicity'].value_counts()
sns.scatterplot(x='reading score',y='math score',hue='race/ethnicity',data=train)

sns.scatterplot(x='reading score',y='writing score',hue='race/ethnicity',data=train)

sns.scatterplot(x='math score',y='writing score',hue='race/ethnicity',data=train)

sns.countplot(x='race/ethnicity',hue='gender',data=train)
plt.savefig('f8')
train['parental level of education'].value_counts()
sns.countplot(hue='gender',x='parental level of education',data=train)
plt.savefig('f9')
sns.countplot(train['test preparation course'],hue='parental level of education',data=train)
plt.savefig('f10')
sns.scatterplot(x='math score',y='writing score',hue='test preparation course',data=train)
plt.savefig('f11')
sns.scatterplot(x='reading score',y='writing score',hue='test preparation course',data=train)
plt.savefig('f12')
sns.scatterplot(x='math score',y='reading score',hue='test preparation course',data=train)
plt.savefig('f13')
sns.jointplot(x='math score',y='reading score',kind='reg',data=train)
plt.savefig('f14')
sns.jointplot(x='math score',y='reading score',kind='hex',data=train)
plt.savefig('f15')
plt.figure(figsize=(10,10))
sns.heatmap(train.corr())
plt.savefig('f16')
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
col=[col for col in train.columns if train[col].dtype=='O']
for c in col:
    train[c]=encoder.fit_transform(train[c])
    
train.head()
train=train.drop_duplicates()
plt.figure(figsize=(10,10))
sns.heatmap(train.corr())
plt.savefig('f17')
sns.swarmplot(x='gender',y='math score',data=train)
plt.savefig('f19')
sns.swarmplot(x='parental level of education',y='reading score',data=train)
plt.savefig('f20')
sns.swarmplot(x='race/ethnicity',y='writing score',data=train)
plt.savefig('f21')
sns.swarmplot(x='test preparation course',y='writing score',data=train)
plt.savefig('f22')
