# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
os.listdir("/kaggle")
os.listdir("/kaggle/input")
os.listdir("/kaggle/input/mental-health-in-tech-survey")
'''Import required libraries for visualization'''
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
%matplotlib inline
directory = "/kaggle/input/mental-health-in-tech-survey/"
dataset_filename = "survey.csv"
df = pd.read_csv(directory + dataset_filename)
df.head()
df.describe()
df.info()
'''We can see above that work-interfere and state have much missing values.'''
# Processing age
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
def age_process(age):
    if age>=0 and age<=100:
        return age
    else:
        return np.nan

df['Age'] = df['Age'].apply(age_process)
(df['Age']<0).sum()
(df['Age']>100).sum()
(df['Age']<60).sum()
df['Age'].isnull().sum()
fig,ax = plt.subplots(figsize=(8, 6))
sns.distplot(df['Age'].dropna(),ax=ax, kde=False, color='#ffa726')
plt.title('Age Distribution')
plt.ylabel("Frequency")
'''Most of the people working tech industry are within range of 25-40. Well it's true that most of the software engineers don't remain in the industry after the age of 40 '''
# Top 10 countries in the data
country_count = df['Country'].unique()
country_count.size
top_ten_country = Counter(df['Country'].dropna().tolist()).most_common(10)
country_idx = [country[0] for country in top_ten_country]
country_val = [country[1] for country in top_ten_country]
fig,ax = plt.subplots(figsize=(8, 6))
sns.barplot(x = country_idx, y = country_val, ax = ax)
plt.title('Top ten country')
plt.xlabel('Country')
plt.ylabel('Count')
ticks = plt.setp(ax.get_xticklabels(), rotation=90)
# Age vs Family-History
df['Age_Group'] = pd.cut(df['Age'].dropna(),
                        [0, 18, 25, 35, 45, 99],
                        labels=['<18','18-24','25-34','35-44','45+'])

fig,ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x = 'Age_Group', hue = 'family_history', ax = ax)
plt.plot('Age vs Family History')
#Age group vs Treatment
fig,ax = plt.subplots(figsize=(8, 6))
sns.countplot(data = df, x = 'Age_Group', hue='treatment')
plt.title('Age Group vs Treatment')
#Age vs No. of Employees
fig,ax = plt.subplots(figsize=(8, 6))
sns.barplot(data = df, x =  df['no_employees'], y = df['Age'], ax = ax)
plt.title('Age Group vs Group size')
plt.xlabel('Group size at Work')
plt.ylabel('Age')
ticks = plt.setp(ax.get_xticklabels(), rotation=90)

# sns.kdeplot(df['no_employees'], df['Age'], ax = ax)
# 

# sns.barplot(x = country_idx, y = country_val, ax = ax)
# plt.title('Top ten country')
# plt.xlabel('Country')
# plt.ylabel('Count')
# ticks = plt.setp(ax.get_xticklabels(), rotation=90)
total = df['no_employees'].dropna().shape[0] * 1.0
employee_count  = Counter(df['no_employees'].dropna().tolist())
for key,val in employee_count.items():
    employee_count[key] = employee_count[key] / total
employee_group = np.asarray(list(employee_count.keys()))
employee_val = np.asarray(list(employee_count.values()))
sns.barplot(x = employee_group , y = employee_val)
plt.title('employee group ratio')
plt.ylabel('ratio')
plt.xlabel('employee group')
fig,ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x='no_employees', hue='tech_company', ax=ax)
ticks = plt.setp(ax.get_xticklabels(),rotation=45)
plt.title('no_employee vs tech_company')
'''In tech companies, developers have to work/lead a small or medium size teams'''
# Remote Work vs employee grp
fig,ax = plt.subplots(figsize=(8, 6))
sns.countplot(data = df, x = 'no_employees', hue = 'remote_work', ax=ax)
ticks = plt.setp(ax.get_xticklabels(), rotation=45)
plt.title('No. Employees vs Remote Work')
