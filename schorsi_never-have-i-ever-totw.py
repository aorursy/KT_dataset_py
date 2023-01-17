import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
full_df = pd.read_csv('/kaggle/input/have-you-ever/responses.csv')

full_df = full_df.drop(['Timestamp'], axis=1) #timestamp isn't necessary for this challenge

full_df.head()
col_list = list(full_df.columns)

col_list#unhide to see the full list of columns referenced in later cells
task_1 = full_df.loc[full_df['Age range'] == '19-24']

task_1 = task_1.loc[task_1['broken a bone'] == True]

task_1.shape
task_2 = full_df.loc[full_df['ran a marathon'] == True]

task_2 = task_2.loc[task_2['had braces'] == True]

task_2.shape
print('###Full Data Set###\n')

for val in col_list:

    _ = full_df[val].value_counts()

    if val == 'Age range':

        pass

    elif _[1] <= 300 or _[1] >= 3000:

        print(val, '\nFalse:', _[0], 'True:', _[1],'\n')



print('\n\n###Task 1 Subset###\n')

for val in col_list:

    _ = task_1[val].value_counts()

    if val == 'Age range' or val == 'broken a bone':

        pass

    elif _[1] <= 43 or _[1] >= 387:

        print(val, '\nFalse:', _[0], 'True:', _[1],'\n')



print('\n\n###Task 2 Subset###\n')

for val in col_list:

    _ = task_2[val].value_counts()

    if val == 'ran a marathon' or val == 'had braces':

        pass

    elif _[1] <= 10 or _[1] >= 63:

        print(val, '\nFalse:', _[0], 'True:', _[1],'\n')
import matplotlib.pyplot as plt

import seaborn as sns
for col in col_list[1:]:

    fig, ax =plt.subplots(1,3,  figsize=(10, 5))

    ax[0].set_title('Full Data Frame')

    sns.countplot(full_df[col], ax=ax[0])

    ax[1].set_title('Task 1 subset')

    sns.countplot(task_1[col], ax=ax[1])

    ax[2].set_title('Task 2 subset')

    sns.countplot(task_2[col], ax=ax[2])
hypo = full_df.loc[full_df['Age range'] != '19-24']

hypo = hypo.loc[hypo['Age range'] != '0-18']

hypo2 = hypo.loc[hypo['broken a bone'] == True]

hypo = hypo.loc[hypo['broken a bone'] == False]

print('Cases where person has been 19-24 and broken a bone: ',hypo2.shape[0],'\nCases where a person has been 19-24 but not broken a bone: ', hypo.shape[0])
full_df['Age range'].value_counts()
a18 = full_df.loc[full_df['Age range'] == '0-18']

a24 = full_df.loc[full_df['Age range'] == '19-24']

a34 = full_df.loc[full_df['Age range'] == '25-34']

a44 = full_df.loc[full_df['Age range'] == '35-44']

a54 = full_df.loc[full_df['Age range'] == '45-54']

a64 = full_df.loc[full_df['Age range'] == '55-64']

a74 = full_df.loc[full_df['Age range'] == '65-74']

a100 = full_df.loc[full_df['Age range'] == '75+']
col = 'broken a bone'



fig, ax =plt.subplots(1,8, figsize=(24, 10))

ax[0].set_title('0-18')

sns.countplot(a18[col], ax=ax[0])

ax[1].set_title('19-24')

sns.countplot(a24[col], ax=ax[1])

ax[2].set_title('25-34')

sns.countplot(a34[col], ax=ax[2])

ax[3].set_title('35-44')

sns.countplot(a44[col], ax=ax[3])

ax[4].set_title('45-54')

sns.countplot(a54[col], ax=ax[4])

ax[5].set_title('55-64')

sns.countplot(a64[col], ax=ax[5])

ax[6].set_title('65-74')

sns.countplot(a74[col], ax=ax[6])

ax[7].set_title('75+')

sns.countplot(a100[col], ax=ax[7])
