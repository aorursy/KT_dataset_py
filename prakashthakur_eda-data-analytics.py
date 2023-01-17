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
data=pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')
data.head()
data=data.iloc[:, 1:]
data.isnull().sum()
data=data.dropna(axis='rows')
data.isnull().sum()
data.shape
data.columns
data.columns=['Job_Title', 'Salary_Estimate', 'Job_Description',

       'Rating', 'Company_Name', 'Location', 'Headquarters', 'Size', 'Founded',

       'Type_of_ownership', 'Industry', 'Sector', 'Revenue', 'Competitors',

       'Easy Apply']
data.head()
import matplotlib.pyplot as plt

import seaborn as sns

plt.Figure(figsize=(10,8))
data['Rating'].value_counts()
data['Rating'].describe()
data['Rating'].median()
data['Rating'].mode()
sns.distplot(data['Rating'])
sns.boxplot(data['Rating'], color='red')
data['Job_Title'].value_counts()
data['Salary_Estimate'].shape
data['Salary_Estimate'].value_counts()
data['Company_Name'].value_counts()
data['Location'].value_counts()
data['Headquarters'].value_counts()
data['Size'].value_counts()
size=sns.countplot(x='Size', data= data)

size.set_xticklabels(size.get_xticklabels(), rotation=90)
data['Founded'].value_counts()
data['Type_of_ownership'].value_counts()
ownership=sns.countplot(x='Type_of_ownership', data=data)

ownership.set_xticklabels(ownership.get_xticklabels(), rotation=90)
data['Industry'].value_counts()
data['Sector'].value_counts()
chart=sns.countplot(x='Sector', data=data)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
data['Revenue'].value_counts()
Rev=sns.countplot(x='Revenue', data=data)

Rev.set_xticklabels(Rev.get_xticklabels(), rotation=90)
data['Competitors'].value_counts()
data['Easy Apply'].value_counts()
data_ind=data
data_ind.head()
data_ind.set_index('Founded', inplace=True)
data_ind.head()
data_ind.loc[2000].Company_Name
data_ind.loc[2000].Rating.max()
data_ind.loc[2000].Rating.min()
data_ind.loc[2008].Salary_Estimate.max()
data_ind.loc[2008].Salary_Estimate.min()
sns.relplot(x="Rating", y="Salary_Estimate", alpha=.5, palette="muted",

            height=6, data=data_ind)