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
import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.head()
df.drop(['sl_no'], axis = 1, inplace = True)
plt.figure(figsize = (10,5))

sns.heatmap(df.corr(), cmap = 'YlGnBu', annot = True);
print('Have a look at descriptive statistics of the data.')

df.describe()
plt.figure(figsize = (10,8))

sns.countplot(df['hsc_s'], hue = df['specialisation']);
plt.figure(figsize = (10,8))

sns.countplot(df['hsc_s'], hue = df['degree_t']);
sns.violinplot(df['specialisation'], df['salary'])

plt.ylabel('Salary in 10 lakhs')

df.pivot_table(values='salary', columns=['specialisation'], aggfunc=np.mean)
print('Mean salries according to the undergraduate dergees.')

df.pivot_table(values='salary', columns=['degree_t'], aggfunc=np.mean)
commerce = df[df['hsc_s'] == 'Commerce']
plt.figure(figsize = (8,4))

sns.countplot(commerce['gender'])
print('There is not too much difference between the mean percentages of the males and females except the percentage in mba degree')

commerce.pivot_table(values=['ssc_p', 'hsc_p', 'degree_p', 'mba_p'], columns=['gender'], aggfunc=np.mean)
sns.countplot(commerce['gender'], hue = commerce['status'])
science = df[df['hsc_s'] == 'Science']

sns.countplot(science['gender'])
print('Females of Science stream have done well then males except in high school exams.')

science.pivot_table(values=['ssc_p', 'hsc_p', 'degree_p', 'mba_p'], columns=['gender'], aggfunc=np.mean)
sns.countplot(science['gender'], hue = science['status'])