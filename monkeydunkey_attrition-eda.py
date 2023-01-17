import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt



%matplotlib inline
data = pd.read_csv('../input/HR_comma_sep.csv')

data.head(15)
plt.figure(figsize = (12,9))

sns.countplot(hue="salary", data = data.sort_values(by = 'sales')[['salary', 'sales']], x = 'sales')
plt.figure(figsize = (12,9))

sns.countplot(hue="salary", data = data[data.left == 1].sort_values(by = 'sales')[['salary', 'sales']], x = 'sales')
temp = data[['salary', 'sales', 'left']]

temp_Remain = temp[temp.left == 0][['salary', 'sales']].groupby(['sales', 'salary']).size()

temp_left = temp[temp.left == 1][['salary', 'sales']].groupby(['sales', 'salary']).size()

temp = (temp_left/(temp_Remain+temp_left))*100

temp = temp.reset_index()

temp.columns = ['Dept', 'Salary', 'Attrition Rate']

temp.Salary = temp.Salary.map({'high' : 2, 'medium' : 1, 'low' : 0})

temp = temp.sort_values(by = ['Dept','Salary'])

temp.Salary = temp.Salary.map({2 : 'high', 1 : 'medium', 0 : 'low'})

plt.figure(figsize = (12,9))

sns.barplot(x="Dept", y="Attrition Rate", hue="Salary", data=temp)