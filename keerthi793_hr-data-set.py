# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
data = pd.read_csv('../input/HR_comma_sep.csv')
data.info()
data.head()
print('Employee stayed : ',data[data['left'] == 0]['left'].value_counts())

print('Employee left : ',data[data['left'] == 1]['left'].value_counts())

fig = plt.figure(figsize = (8,4))

sns.countplot(data['promotion_last_5years'], hue=data['left'], palette='Paired')

plt.suptitle('WHETHER PROMOTED IN THE LAST FIVE YEARS', fontsize=14, fontweight='bold')

plt.legend(title='Left')

plt.xlabel('Promotion in last 5years')

plt.ylabel('Number of employees')

fig = plt.figure(figsize = (8,4))

sns.countplot(data['salary'], hue=data['left'], palette='coolwarm')

plt.suptitle('SALARY LEVEL', fontsize=14, fontweight='bold')

plt.legend(title='Left')

plt.xlabel('Salary offered')

plt.ylabel('Number of employees')
fig = plt.figure(figsize = (8,4))

sns.countplot(data['Work_accident'], hue=data['left'], palette='Paired')

plt.suptitle('HAD ANY WORK ACCIDENT', fontsize=14, fontweight='bold')

plt.legend(title='Left')

plt.xlabel('Work accident')

plt.ylabel('Number of employees')
sns.lmplot(x='last_evaluation', y='average_montly_hours', data=data, hue='left', fit_reg=False)
sns.lmplot(x='last_evaluation', y='average_montly_hours', data=data, col='left', hue='promotion_last_5years', fit_reg=False)
sns.lmplot(x='number_project', y='average_montly_hours', data=data, col='left', hue='promotion_last_5years', fit_reg=False, palette='coolwarm')
sns.lmplot(x='number_project', y='average_montly_hours', data=data, col='left', hue='salary', fit_reg=False, palette='coolwarm')
sns.lmplot(x='time_spend_company', y='average_montly_hours', data=data, col='left', hue='promotion_last_5years', fit_reg=False)
sns.lmplot(x='time_spend_company', y='average_montly_hours', data=data, col='left', hue='salary', fit_reg=False)
empLeft = data[data['left'] == 0]

empStayed = data[data['left'] == 1]
fig = plt.figure(figsize = (10,8))

plt.hist(empLeft['satisfaction_level'], bins=20, alpha = 0.7, label='Employee Left',color='black')

plt.hist(empStayed['satisfaction_level'],bins=20,alpha = 0.2, label='Employee Stayed',color='blue')

plt.legend()

plt.xlabel('Employees satifaction level')

#plt.ylabel('No of employees')
fig = plt.figure(figsize=(12,6))

sns.countplot(data['sales'], hue=data['left'])