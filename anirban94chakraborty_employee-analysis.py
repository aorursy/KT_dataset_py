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
data = pd.read_csv('/kaggle/input/employee-analysis/employee.csv')
data.shape
data.columns
data.head()
data.tail()
data.describe()
data.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns
sns.distplot(data['Age'])

plt.plot()
data['EducationField'].value_counts().plot(kind='bar')
data['EducationField'].value_counts().plot(kind='line')
data['EducationField'].value_counts().plot(kind='area')
data['EducationField'].value_counts().plot(kind='pie')
plt.rcParams['figure.figsize'] = (12,5)

plt.style.use('fivethirtyeight')

sns.lineplot(data['EducationField'], data['Age'])



plt.title('Average Age of Employees from each Department')

plt.show()
avg_sal = data['MonthlyIncome'].mean()

avg_sal
emp_over_40 = data[data['Age'] > 40]

emp_over_40
emp_over_40 
emp_over_40[emp_over_40['MonthlyIncome'] < avg_sal][['MonthlyIncome', 'Age']].shape
data['Attrition'] = data['Attrition'].replace(('Yes','No'),(1,0))
data[['EducationField', 'Attrition']].groupby(['EducationField']).agg(['count','sum', 'mean'])
data['EducationField'].value_counts()
pd.crosstab(data['Department'], data['BusinessTravel'])
pd.crosstab(data['Department'], data['BusinessTravel']).plot(kind='bar')
data[['BusinessTravel','MonthlyIncome']].groupby('BusinessTravel').agg(['min','mean','max']).style.background_gradient(cmap='copper')
sns.boxplot(data['Department'], data['PercentSalaryHike'])
sns.lineplot(y = data['MonthlyIncome'], x = data['PerformanceRating'])