import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
data.head(10)
data.shape
#replacing space with underscore('_') for convenient accessing of columns

data.columns = data.columns.str.replace(' ','_')
data.columns
# checking for null values

data.isna().sum()
sns.set(font_scale=1.4)

plt.figure(figsize=(7,6))

sns.countplot(x='gender',data=data)
print(data.gender.value_counts())

print('----------------------------------')

print(data.gender.value_counts(normalize=True))
sns.set(font_scale=1.4)

plt.figure(figsize=(7,6))

sns.countplot(x='race/ethnicity',data=data)
sns.set(font_scale=1.4)

plt.figure(figsize=(14,8))

sns.countplot(x='parental_level_of_education',data=data)
sns.set(font_scale=1.4)

plt.figure(figsize=(7,6))

sns.countplot(x='lunch',data=data)
sns.set(font_scale=1.4)

plt.figure(figsize=(7,6))

sns.countplot(x='test_preparation_course',data=data)
plt.figure(figsize=(18,8))

sns.countplot(x="parental_level_of_education",hue="gender",data=data)

plt.ylabel('Count of Parental Level of Education')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data.groupby('gender').agg('mean').plot(kind='bar',figsize=(7,5.5))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data.groupby('parental_level_of_education').agg('mean').plot(kind='barh',figsize=(10,10))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data.groupby('test_preparation_course').agg('mean').plot(kind='barh',figsize=(7,7))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data.groupby('lunch').agg('mean').plot(kind='barh',figsize=(7,7))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data.groupby('race/ethnicity').agg('mean').plot(kind='barh',figsize=(9,9))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.figure(figsize=(8,5.5))

sns.countplot(x='race/ethnicity',hue='test_preparation_course',data=data)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x='gender',hue='lunch',data=data)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x='gender',hue='test_preparation_course',data=data)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data.groupby(['race/ethnicity','gender']).agg('mean').plot(kind='bar',figsize=(12,8))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data.groupby(['gender','lunch']).agg('mean').plot(kind='bar',figsize=(12,8))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data.groupby(['gender','parental_level_of_education']).agg('mean').plot(kind='barh',figsize=(12,10))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data.groupby(['gender','test_preparation_course']).agg('mean').plot(kind='bar',figsize=(12,8))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data.groupby(['race/ethnicity','parental_level_of_education']).agg('mean').plot(kind='barh',figsize=(12,12))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)