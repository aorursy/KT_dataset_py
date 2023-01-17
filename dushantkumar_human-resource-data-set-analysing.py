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
data=pd.read_csv('/kaggle/input/human-resources-data-set/HRDataset_v13.csv')
data.shape
data.head()
data.describe()
data.isnull()
data['Sex'].unique()
data['Sex'].replace('F','female',inplace=True)
data['Sex'].dropna(inplace=True)
data['Sex'].isnull()

data['Sex'].replace('M','Male',inplace=True)
data['Sex'].unique()
data['Sex'].value_counts()
data['Sex'].value_counts().plot(kind='bar')
import seaborn as sns

import matplotlib.pyplot as plt

# plot through sns
ax=sns.countplot(data['Sex'])
# Gender diversity across departmets

plt.figure(figsize=(16,9))

ax=sns.countplot(data['Department'],hue=data['Sex'])
#Conclusions from graph :



#No males in executive office and no females in software engineering department.

#Gender diversity is not maintained in production department and software engineering.

#No.of females is nearly double the number of males
plt.figure(figsize=(10,6))

data['MaritalDesc'].value_counts().plot(kind='pie')
data['CitizenDesc'].unique()
data['CitizenDesc'].value_counts().plot(kind='bar')
data['Position'].value_counts()
plt.figure(figsize=(20,12))

data['Position'].value_counts().plot(kind='bar')
data['PerformanceScore'].unique()
data['PerformanceScore'].dropna(inplace=True)
data['PerformanceScore'].value_counts().plot(kind='bar')
df_perf = pd.get_dummies(data,columns=['PerformanceScore'])
df_perf.head()
data['PerformanceScore'].unique()
col_plot= [col for col in df_perf if col.startswith('Performance')]

col_plot
fig, axes = plt.subplots(2, 2, figsize=(16,9))

for i,j in enumerate(col_plot):

    df_perf.plot(x=j,y='PayRate',ax = axes.flat[i],kind='scatter')
data['ManagerName'].unique()
data['ManagerName'].dropna(inplace=True)
plt.figure(figsize=(16,20))

sns.countplot(y=data['ManagerName'],hue=data['PerformanceScore'])


plt.figure(figsize=(16,9))

data.groupby('Department')['PayRate'].sum().plot(kind='bar')
plt.figure(figsize=(16,9))

data.groupby('Position')['PayRate'].sum().plot(kind='bar')
data.columns
data.loc[data['PayRate'].idxmax()]
data.loc[data['PayRate'].idxmin()]