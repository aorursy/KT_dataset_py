# Import necessary packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the data into memory

df = pd.read_csv('../input/income-classification/income_evaluation.csv')
# Let's take a look at our data

df.head()
df.tail()
# 32k rows of data and we have 15 variables

df.shape
df.info()
df.isnull().any()
df = df.rename(columns={'age': 'age',

                         ' workclass': 'workclass',

                         ' fnlwgt': 'final_weight',

                         ' education': 'education',

                         ' education-num': 'education_num',

                         ' marital-status': 'marital_status',

                         ' occupation': 'occupation',

                         ' relationship': 'relationship',

                         ' race': 'race',

                         ' sex': 'sex',

                         ' capital-gain': 'capital_gain',

                         ' capital-loss': 'capital_loss',

                         ' hours-per-week': 'hrs_per_week',

                         ' native-country': 'native_country',

                         ' income': 'income'

                        })

df.columns
df['income'].unique()
df['income_encoded'] = [1 if value == ' >50K' else 0 for value in df['income'].values]

df['income_encoded'].unique()
# look better now

df.sample(5)
# Let's check some descriptive statistics

df.describe()
# create a blank canvas

plt.figure(figsize=(10, 10))

sns.heatmap(df.corr(), annot=True, fmt='.2f')

plt.title('Overview heatmap')
plt.figure(figsize=(10, 10))

sns.boxplot('income', 'education_num', data=df)
plt.figure(figsize=(10, 10))

sns.boxplot('income', 'age', data=df)
plt.figure(figsize=(10, 10))

sns.boxplot('income', 'hrs_per_week', data=df)
df['education'].value_counts().to_frame()
plt.figure(figsize=(20, 10))

sns.countplot('education', hue='income', data=df)
df_filtered = df.isin({'education': ["HS-grad", "Some-college"]})

df_filtered.any()

#plt.figure(figsize=(20, 10))

#sns.countplot('education', hue='income', data=df[df_filtered])
df['occupation'].value_counts().head(3)
df[df['income'] == ' >50K']['occupation'].value_counts().head(3)
pd.crosstab(df["occupation"], df['income']).plot(kind='barh', stacked=True, figsize=(20, 10))