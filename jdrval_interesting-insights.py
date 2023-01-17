import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np # Linear algebra

import pandas as pd # Data processing

import missingno # # Missing values

import seaborn as sns # Data visualization

sns.set()
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
test_df = pd.read_csv('../input/test-data/Data Analyst - Test Data - US.csv') # Kaggle

#test_df = pd.read_csv('Data Analyst - Test Data - US.csv')

print('Reading: success')
test_df.info()
missingno.matrix(test_df, figsize=(20,7))
test_df.head()
test_df.tail()
test_df.describe()
test_df['Location'].value_counts()
test_df['Review'].isnull().value_counts()
def unique_val(columns, data):

    for column in columns:

        unique_values = data[column].unique()

        print(unique_values)



unique_val(['Location', 'Review'], test_df)
def null_val(columns, data):

    for column in columns:

        x = test_df[column].isnull().value_counts()

        print(column)

        print(x)



null_val(['Location', 'Review', 'date'], test_df)
test_df['date'].value_counts()
# Number of unique values per column

def null_val(columns, data):

    for column in columns:

        x = test_df[column].value_counts().count()

        print(column)

        print(x)



null_val(['Location', 'Review', 'date'], test_df)
test_df.duplicated().value_counts()
test_df[test_df.duplicated() == True]
sns.set(font_scale=2)

sns.set_style("white")

plt.figure(figsize=(35,10))

plt.title("Quantity of reviews per location")

sns.barplot(x=test_df['Location'].value_counts()[0:10].index, y=test_df['Location'].value_counts()[0:10])

plt.ylabel("Number of reviews")

plt.xlabel("Countries", labelpad=14)
sns.set(font_scale=2)

sns.set_style("white")

plt.figure(figsize=(35,10))

plt.title("Quantity of reviews per date")

sns.barplot(x=test_df['date'].value_counts()[0:10].index, y=test_df['date'].value_counts()[0:10])

plt.ylabel("Number of reviews")

plt.xlabel("Dates", labelpad=14)