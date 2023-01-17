import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')

print(data.info())

data.head()
null_values = data.isnull().sum()

plt.figure(figsize = (20,10))

sns.barplot(null_values.index, null_values, color = 'grey')

plt.suptitle('Missing values in the dataset', size = 20)

plt.ylabel('Missing values')

plt.xticks(rotation = '45')

plt.show()
categorical_columns = []

for col in data.columns:

    print(f'Unique rows in {col}:', data[col].nunique())

    if data[col].nunique() < 15:

        categorical_columns.append(col)

print('Categorical columns:',categorical_columns)
for col in list(set(data.columns) - set(categorical_columns)):

    if data[col].dtypes == 'object':

        data[col] = data[col].fillna('')

print(data.isnull().sum())
data['employment_type'] = data['employment_type'].fillna('Employement_Unavailable')

data['required_experience'] = data['required_experience'].fillna('Experience_Unavailable')

data['required_education'] = data['required_education'].fillna('Education_Unavailable')
fig, axs = plt.subplots(len(categorical_columns)//2 , 2, figsize = (25,30))

plt_row = 0

plt_col = 0

for i, col in enumerate(categorical_columns):

    if col == 'fraudulent':

        continue

    _ = data[col].value_counts()

    sns.barplot(_.index, _, ax = axs[plt_row][plt_col])

    axs[plt_row][plt_col].set_title(f'Distribution of {col}', size = 20)

    for tick in axs[plt_row,plt_col].get_xticklabels():

        tick.set_rotation(30)

    if plt_col == 0:

        plt_col = 1

    else:

        plt_col = 0

        plt_row += 1

plt.show()
figure = plt.figure(figsize = (15,7))

target_dist = data['fraudulent'].value_counts()

sns.barplot(target_dist.index, target_dist)

plt.title('Target Distribution', size = 18)

plt.show()
data.drop('job_id', axis = 1, inplace = True)

data.head()