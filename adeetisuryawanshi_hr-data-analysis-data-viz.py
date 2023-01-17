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
df = pd.read_csv('../input/human-resources-data-set/HRDataset_v13.csv')
df.head(5)
df.describe()
df.shape
df.dropna(axis = 0, how = 'all',inplace = True)
df.shape
df.info()
df.describe()
df.isnull().sum()
df.drop(columns = ['Employee_Name','MaritalStatusID','HispanicLatino','Zip','Position','MarriedID','Termd','GenderID'], inplace = True)
import seaborn as sns

sns.countplot(x = 'Sex', data = df)
def calc_age(entity):

    current_year = pd.to_datetime('today').year

    birth_year = pd.to_datetime(entity).year

    if birth_year>current_year:

        birth_year = birth_year-100

    return current_year - birth_year
df['age'] = df['DOB'].map(calc_age)
sns.distplot(a=df['age'],kde = False);
import matplotlib.pyplot as plt

plt.figure(figsize=(100,45))

sns.set(font_scale=5)

sns.countplot(x='Department',data=df,hue='Sex');
df['PerformanceScore'].value_counts()
grouped_data = df.groupby(['ManagerName','PerformanceScore']).size().reset_index()

grouped_data.columns = ['ManagerName','PerformanceScore','Count']
grouped_data=grouped_data.pivot(columns='PerformanceScore',index = 'ManagerName', values = 'Count')
grouped_data.plot(kind ='bar',stacked = True,figsize=(60,40))
plt.figure(figsize=(16,6.5))

sns.set(font_scale=2)

sns.regplot(x='EmpSatisfaction',y='PayRate',data = df);
df['MaritalDesc'].value_counts()
df['MaritalDesc'].value_counts().plot(kind='pie')
grouped_data = df.groupby(['MaritalDesc', 'PerformanceScore']).size().reset_index()

grouped_data.columns = ['MaritalDesc','PerformanceScore','Count']

grouped_data=grouped_data.pivot(columns='PerformanceScore', index='MaritalDesc', values='Count')
grouped_data.plot(kind = 'bar',stacked=True,figsize=(30,14),)
plt.figure(figsize=(15,20))

sns.set(font_scale=2)

sns.countplot(y = 'RecruitmentSource',data = df,order = df['RecruitmentSource'].value_counts().index)