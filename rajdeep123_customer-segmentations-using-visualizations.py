# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
customers = pd.read_csv('../input/Mall_Customers.csv')

customers.head()
customers.rename(columns={'Genre': 'Gender'},inplace=True)
customers.describe()

#customers.head()
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(x='Gender', data=customers);

plt.title('Distribution of Gender');
customers.hist('Age', bins=35);

plt.title('Distribution of Age');

plt.xlabel('Age');
plt.hist('Age', data=customers[customers['Gender'] == 'Male'], alpha=0.5, label='Male');

plt.hist('Age', data=customers[customers['Gender'] == 'Female'], alpha=0.5, label='Female');

plt.title('Distribution of Age by Gender');

plt.xlabel('Age');

plt.legend();
customers.hist('Annual Income (k$)');

plt.title('Annual Income Distribution in Thousands of Dollars');

plt.xlabel('Thousands of Dollars');
plt.hist('Annual Income (k$)', data=customers[customers['Gender'] == 'Male'], alpha=0.5, label='Male');

plt.hist('Annual Income (k$)', data=customers[customers['Gender'] == 'Female'], alpha=0.5, label='Female');

plt.title('Distribution of Income by Gender');

plt.xlabel('Income (Thousands of Dollars)');

plt.legend();
male_customers = customers[customers['Gender'] == 'Male']

female_customers = customers[customers['Gender'] == 'Female']





print(male_customers['Spending Score (1-100)'].mean())

print(female_customers['Spending Score (1-100)'].mean())
sns.scatterplot('Age', 'Annual Income (k$)', hue='Gender', data=customers);

plt.title('Age to Income, Colored by Gender');
sns.heatmap(customers.corr(), annot=True)
sns.scatterplot('Age', 'Spending Score (1-100)', hue='Gender', data=customers);

plt.title('Age to Spending Score, Colored by Gender');
sns.heatmap(female_customers.corr(), annot=True);

plt.title('Correlation Heatmap - Female');
sns.heatmap(male_customers.corr(), annot=True);

plt.title('Correlation Heatmap - Female');
sns.lmplot('Age', 'Spending Score (1-100)', data=female_customers);

plt.title('Age to Spending Score, Female Only');
sns.scatterplot('Annual Income (k$)', 'Spending Score (1-100)', hue='Gender', data=customers);

plt.title('Annual Income to Spending Score, Colored by Gender');