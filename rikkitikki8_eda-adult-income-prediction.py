# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv('../input/adult.csv')

df.head()
len(df)
def outliers_indices(feature):

    mid = df[feature].mean()

    sigma = df[feature].std()

    return df[(df[feature] < mid - 3*sigma) | (df[feature] > mid + 3*sigma)].index
sns.boxplot(df['age']);
wrong_age = outliers_indices('age')

df.drop(wrong_age, inplace=True)

len(wrong_age)
plt.figure(figsize=(10, 20))

sns.countplot(y='age', hue='income', data=df);
df_less_50 = df[df['income']=='<=50K']

print('mean = ', df_less_50['age'].mean())

print('std = ', df_less_50['age'].std())

df_less_50['age'].hist(bins=10);
df_more_50 = df[df['income']=='>50K']

print('mean = ', df_more_50['age'].mean())

print('std = ', df_more_50['age'].std())

df_more_50['age'].hist(bins=10);
df_more_50['complete_secondary'] = df_more_50['education'].apply(

    lambda x: True if x in ('Bachelors', 'Prof-school', 

                            'Assoc-acdm','Assoc-voc', 

                            'Masters', 'Doctorate') else False)

count_complete_secondary = df_more_50.groupby('complete_secondary')['age'].count()



print(int(round(count_complete_secondary[True] / len(df_more_50) * 100)), 

      '% of people who earn >50K have completed secondary education\n')



count_complete_secondary
print(df['marital-status'].value_counts())
df['married'] = df['marital-status'].apply(lambda x: 1 if x in (

    'Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse') else 0)

df['num_income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
df_male = df[df['gender'] ==  'Male']

# pd.crosstab(df_male['num_income'], df_male['married'])



print(df_male.groupby('married')['num_income'].sum() / df_male.groupby('married')['num_income'].count() * 100)

print('\nт.е, 7% неженатых мужчин зарабатывают >50K')
df['hours-per-week'].max()
df_hours_99 = df[df['hours-per-week'] == 99]

len(df_hours_99)
print(int(round(df_hours_99[df_hours_99['income'] == '>50K'].age.count() / len(df_hours_99) * 100)), '%')
sns.boxplot(df['hours-per-week']);
wrong_hours = outliers_indices('hours-per-week')

df.drop(wrong_hours, inplace=True)

len(wrong_hours)
df.pivot_table(values=['hours-per-week'], index=['income'], aggfunc='mean')
sns.boxplot(df['income'], df['hours-per-week'])
from scipy.stats import pointbiserialr

r = pointbiserialr(df['hours-per-week'], df['num_income'])

print('Point-biserial correlation:', r[0], 'p-value:', r[1])
from scipy.stats import spearmanr, kendalltau
r = spearmanr(df['educational-num'], df['hours-per-week'])

print('Spearman correlation:', r[0], 'p-value:', r[1])
r = kendalltau(df['educational-num'], df['hours-per-week'])

print("Kendall's correlation:", r[0], 'p-value:', r[1])
df.groupby('educational-num')['hours-per-week'].mean().plot(kind='bar') 

plt.ylabel('hours-per-week')

plt.show();
country_df = df[['native-country', 'income']]

country_df['more_50K'] = country_df['income'].apply(lambda x: 1 if x == '>50K' else 0)

country_df['less_50K'] = country_df['income'].apply(lambda x: 1 if x == '<=50K' else 0)

country_df.drop(['income'], axis=1, inplace=True)



country_without_America = country_df[country_df['native-country'] != 'United-States']

country_without_America.groupby('native-country').sum().plot(kind='bar', rot=90, figsize=(20, 10));
country_America = country_df[country_df['native-country'] == 'United-States']

country_America.groupby('native-country').sum().plot(kind='bar', rot=0, figsize=(7, 7));
df_without_America = df[df['native-country'] != 'United-States']

plt.figure(figsize=(15, 50))

sns.countplot(y='native-country', hue='income', data=df_without_America);
df_America = df[df['native-country'] == 'United-States']

plt.figure(figsize=(5, 5))

sns.countplot(y='native-country', hue='income', data=df_America);
new_values_gender = {'Male':0 , 'Female':1} 

df['gender-num'] = df['gender'].map(new_values_gender)
df.head().T
plt.figure(figsize=(10, 10))

sns.heatmap(df.corr());
df['native-country'].unique()
country_df = df[df['native-country'] == 'Japan']

sns.countplot(x='race', data=country_df);
df_table = df.pivot_table(

    values='num_income',

    index='native-country',

    columns='gender', aggfunc=np.mean).fillna(0).applymap(float)

sns.heatmap(df_table);
df[(df['gender'] == 'Male') & (df['relationship'] == 'Wife')].index
df[(df['gender'] == 'Female') & (df['relationship'] == 'Husband')].index
df.drop([5661, 16856, 43422, 23390], inplace=True)
df_table = df.pivot_table(

    values='num_income',

    index='relationship',

    columns='gender', aggfunc=np.mean).fillna(0).applymap(float)

sns.heatmap(df_table);