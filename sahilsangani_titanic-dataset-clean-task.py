# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/titanic/Titanic.csv")
df
df.describe()
df.info()
df.isnull().sum()
df['Age'] = df['Age'].fillna(value = df['Age'].mean())
df['Age'].isnull().sum()
df['Cabin'].head()
df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True, errors='ignore') 
# inplace make changes in a dataframe
# errors='ignore'--> it will ignore the error if columns doesn't exists.
df
df = pd.get_dummies(df,drop_first=True)
# Here, drop_first don't uses redudant feature.
# Below is example for the same.
df.head(3)
df.isnull().sum()
sum(n < 0 for n in df['Age'].values.flatten())
# Checking for negative values in Age column
print(max(df['Age'].values.flatten()))
print(min(df['Age'].values.flatten()))
# Checking for maximum and minimum Age in dataset
df.boxplot(figsize=(20,10))
# Box-plot can apply only on dataframe (Can't apply on series like df['Age'].boxplot())
# df.boxplot(column='Age', return_type='axes', figsize=(5,8));
ax = sns.boxplot(x=df["Age"])
df['Age'].describe()
df['Age'].skew()
IQR_AGE = df['Age'].quantile(0.75) - df['Age'].quantile(0.25)
IQR_AGE
# We have only upper outliers for Age; We will work only on upper outlier.
Upper_Outlier_Limit = df['Age'].quantile(0.75) + 1.5*IQR_AGE
Upper_Outlier_Limit
Upper_Outlier_Values = df[(df['Age']>Upper_Outlier_Limit)]
Upper_Outlier_Values['Age'].count() # 42 outlier values are found.
df['Age'] = np.where(df['Age']>Upper_Outlier_Limit,df['Age'].quantile(0.95),df['Age'])
df
Upper_Outlier_Values = df[(df['Age']>Upper_Outlier_Limit)]
print("Number of outliers: {}".format(Upper_Outlier_Values['Age'].count()))
print("Number of negative values for Age: {}".format(sum(n < 0 for n in df['Age'].values.flatten())))
print("Skewness of Age data: {}".format(df['Age'].skew()))
ax = sns.boxplot(x=df["Age"])

IQR_FARE = df['Fare'].quantile(0.75) - df['Fare'].quantile(0.25)
print("IQR Fare: {}".format(IQR_FARE))
Upper_Outlier_Limit_F = df['Fare'].quantile(0.75) + 1.5*IQR_AGE
print("Upper_Outlier_Limit_F: {}".format(Upper_Outlier_Limit_F))
Upper_Outlier_Values_F = df[(df['Fare']>Upper_Outlier_Limit_F)]
print("Upper_Outlier_Values_F: {}".format(Upper_Outlier_Values_F['Fare'].count()))
ax_F = sns.boxplot(x=df["Fare"])
print("Skewness of Fare data: {}".format(df['Fare'].skew()))
# df['Fare'].median()
df['Fare'] = np.where(df['Fare']>Upper_Outlier_Limit_F,df['Fare'].quantile(0.80),df['Fare'])
df

print("Upper_Outlier_Values_F: {}".format(Upper_Outlier_Values_F['Fare'].count()))
ax_F = sns.boxplot(x=df["Fare"])
print("Skewness of Fare data: {}".format(df['Fare'].skew()))
# df['Fare'].median()
df.describe()
df.boxplot(figsize=(20,10))
print(df.isnull().sum())
