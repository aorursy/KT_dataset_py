# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read the csv file

df = pd.read_csv('/kaggle/input/titanic/train.csv')

# df.head()




import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')



age = df['Age']



# find measures of central tendency and dispersion

print(age.describe())



# plot age 

age.plot.box();
first_quantile = 20.125

third_quantile = 38.0



iqr = third_quantile - first_quantile



outlier = age > (third_quantile + (iqr) * 1.5)

percent_outlier = (outlier.sum()/len(df)) * 100



print(f'Outlier percentage: {percent_outlier:.2f}')



pass_age_below_40 = (age < 40).sum()/ len(age)

print(f'Passengers below 40 years of age: {pass_age_below_40}')
bins = list(range(0, 90, 10))

labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '80-90']



age_discretized = pd.cut(df['Age'], bins=bins, labels=labels)

new_df = df[['Survived']].assign(Age=age_discretized.values)



survivors = new_df[new_df.Survived == 1]

non_survivors = new_df[new_df.Survived == 0]



survivors_age = survivors.groupby(['Age']).size().values

non_survivors_age = non_survivors.groupby(['Age']).size().values



totals = survivors_age + non_survivors_age



data1_percentages = (survivors_age/ totals) * 100 

data2_percentages = (non_survivors_age/ totals) *100 



plt.figure(figsize=(10, 5))



plt.bar(range(len(survivors_age)), survivors_age, label='Survivors', alpha=0.5, color='g')

plt.bar(range(len(non_survivors_age)), non_survivors_age, bottom=survivors_age, label='Non-Survivors', alpha=0.5, color='r')



plt.xticks(np.arange(10), labels)

plt.legend(loc='upper right');

# percentage of non survivors by age

print(non_survivors_age/non_survivors_age.sum())
df.head()
missing = df.isnull().sum(axis=0)/len(df) * 100

print(missing)

df.dropna(subset=['Age'], inplace=True)
df.info()
df.drop(columns=['PassengerId'], inplace=True)

df.info()
print(df[['Cabin']].notnull().sum())

df['Cabin'].nunique()
df.drop(columns=['Cabin'], inplace=True)
df.describe()
# normalize Age

age = df['Age']

df['Age'] = age / age.max()

df['Age'].describe()