# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
#Basic statistical Analyis
df.describe().T
#To get a sense of the various data types
df.dtypes
df.info()
#Sorting out the categorical columns
cat_columns = df.select_dtypes(exclude=np.number).columns
#finding the number of unique values in each  numerical column
for i in cat_columns:
    print(f'unique [{i}] count:', df[i].nunique())
#finding the number of null values in the dataset
df.isnull().sum()
#you can see all the null values are in the salary column
#finding percentage of null values
df.isnull().mean()
#visualizing missing data
sns.set_style('whitegrid')
df.isnull().mean().plot.bar(figsize=(12,16))
plt.ylabel('percentage of missing values')
plt.xlabel('Variables')
plt.title('Quantifying missing data')
#you can see the missing data is all in the Salary Column
label_frequency = df['salary'].value_counts()/len(df)

fig = label_frequency.sort_values(ascending=False).plot.bar()
# the red line indicates the limit under which we consider a category rare
fig.axhline(y=0.05, color='red')
fig.set_ylabel('percentage of cars within each category')
fig.set_xlabel('Variable: class')
fig.set_title('Identifying Rare Categories')
plt.show()
#plot the correlation plot
plt.figure(figsize=(12,16))
sns.heatmap(df.corr(),cmap='viridis',annot=True)
#lets plot the number of unique variables
sns.set_style('whitegrid')
df.nunique().plot.bar(figsize=(12,6))
plt.ylabel('Number of unique categories')
plt.xlabel('Variables')
plt.title('Cardinality')
#plots only the numerival variables
df.hist(bins=30,figsize=(12,12),density=True)
sns.pairplot(df)
#finding outliers
def find_boundaries(df, variable, distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary
#finding outliers in the salary Column
upper_boundary, lower_boundary = find_boundaries(df, 'salary', 1.5)
upper_boundary, lower_boundary
#most demande is Mkt&Fin
df['specialisation'].mode()
