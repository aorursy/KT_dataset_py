# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read the CSV and get the data

df_yearly = pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_yearly_variables.csv', delimiter=',')

df_monthly = pd.read_csv('/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv', delimiter=',')

df_yearly.head()
df_monthly.head()
df_yearly.describe()
df_monthly.describe()
df_yearly.dtypes
for column in df_yearly.columns.values.tolist():

    print(column)

    print (df_yearly[column].value_counts())

    print("")   
df_yearly.fillna(0)

missing_data = df_yearly.isnull()

missing_data.head(5)
#Counting missing values in each column:

for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")    
df_yearly.isnull()
#not a good way to clean the string data but doing just to clean it a bit

df_yearly['mean_salary'] = df_yearly['mean_salary'].str.replace(' ', '')

df_yearly['mean_salary'] = df_yearly['mean_salary'].str.replace('_', '')

df_yearly['mean_salary'] = df_yearly['mean_salary'].str.replace('-', '')

df_yearly['mean_salary'] = df_yearly['mean_salary'].str.replace('#', '')
#the following is better way to change the string object to numeric 

df_yearly['mean_salary'].fillna(0)

df_yearly['mean_salary'] = pd.to_numeric(df_yearly['mean_salary'], errors='coerce')

df_yearly['mean_salary'].mean()

#changing the mean salary to float

#df_yearly[["mean_salary"]] = df_yearly[["mean_salary"]].astype("float")

avg_mean_salary = df_yearly["mean_salary"].astype("float").mean(axis=0)

print('The averagemean salary =',avg_mean_salary)

#printing values in all the columns
for value in df_yearly['mean_salary']:

    print("The value is", value)
#finding the correlation among the data

df_yearly.corr()
#let us see the corelation among the following variables

df_yearly[["median_salary","mean_salary","number_of_jobs","no_of_houses"]].corr()
df_yearly[["median_salary","no_of_houses"]].corr()
df_yearly[["mean_salary","no_of_houses"]].corr()
import seaborn as sns

import matplotlib.pyplot as plt

sns.regplot(x="median_salary", y="no_of_houses", data=df_yearly)

plt.ylim(0,)
sns.regplot(x="mean_salary", y="no_of_houses", data=df_yearly)

plt.ylim(0,)