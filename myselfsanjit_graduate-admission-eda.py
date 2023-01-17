# https://www.kaggle.com/biphili/university-admission-in-era-of-nano-degrees

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dataset = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
dataset.head()
dataset.drop(['Serial No.'], axis=1,  inplace=True)
column_names = {'GRE Score': 'gre_score', 'TOEFL Score': 'toefl_score', 'University Rating': 'university_rating', \

                'SOP': 'sop', 'LOR': 'lor', 'CGPA': 'cgpa',\

                'Research': 'research', 'Chance of Admit ': 'chance_of_admit'}
dataset = dataset.rename(columns = column_names)

dataset.head()
dataset.tail()
dataset.shape
dataset.dtypes
for data in dataset.columns:

    print(data)

    print(dataset[data].unique())

    print("="*80)
dataset.describe()
dataset.isnull().any()
plt.subplots(figsize=(10, 5))

sns.heatmap(dataset.corr(), cmap="YlGnBu", annot=True, fmt= '.0%')

plt.show()
plt.subplots(figsize=(10, 5))

dataset.corr().loc['chance_of_admit'].sort_values(ascending=False).plot(kind='bar')
sns.pairplot(dataset, corner=True, diag_kind="kde")
print(f"{dataset['research'].value_counts()/len(dataset)}")

print("="*80)

sns.countplot(dataset['research'])
sns.scatterplot(y="cgpa", x="gre_score", hue="university_rating", data=dataset)
sns.scatterplot(y="cgpa", x="gre_score", hue="research", data=dataset)