%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt 

sns.set(style="whitegrid", color_codes=True)

df = pd.read_csv('../input/retail-marketing/retailMarketingDI.csv')

df.columns = df.columns.str.lower()

print("DataFrame Columns\n", df.columns.tolist())

df.head()
"Number of Rows:%s and Number of Columns:%s" %(len(df), len(df.columns))
print('These are the categorical values')

print("age", df.age.unique().tolist())

print("gender", df.gender.unique().tolist())

print("ownhome", df.ownhome.unique().tolist())

print("married", df.married.unique().tolist())

print("location", df.location.unique().tolist())

print("history", df.history.unique().tolist())

print("catalogs", sorted(df.catalogs.unique().tolist()))

print("childrens", sorted(df.children.unique().tolist()))

print('----------------------------')

print("salary min:%s mean:%s median:%s max:%s mode:%s"% (df.salary.min(), df.salary.mean(),df.salary.median(), df.salary.max(), df.salary.mode()))

print("amount spent min:%s mean:%s median:%s max:%s mode:%s"% (df.amountspent.min(), round(df.amountspent.mean(), 2),df.amountspent.median(), df.amountspent.max(), df.amountspent.mode()))

print("amount spent min:%s mean:%s median:%s max:%s mode:%s"% (df.amountspent.min(), round(df.amountspent.mean(), 2),df.amountspent.median(), df.amountspent.max(), df.amountspent.mode()))

df['age'] = pd.Categorical(df.age, categories=['Young', 'Middle', 'Old'],ordered=True)

df['gender'] = df.gender.astype('category')

df['ownhome'] = df.ownhome.astype('category')

df['married'] = df.married.astype('category')

df['location'] = df.location.astype('category')

df['children'] = df.children.astype('category')

df['history'] = pd.Categorical(df.history, categories=['Low', 'Medium', 'High'],ordered=True)

df['catalogs'] = df.catalogs.astype('category')

print(df.dtypes)
print(100 * df.age.value_counts()/len(df))

print('---------------------')

print(100 * df.gender.value_counts()/len(df))

print('---------------------')

print(100 * df.ownhome.value_counts()/len(df))

print('---------------------')

print(100 * df.married.value_counts()/len(df))

print('---------------------')

print(100 * df.location.value_counts()/len(df))

print('---------------------')

print(100 * df.children.value_counts()/len(df))

print('---------------------')

print(100 * df.history.value_counts()/len(df))

print('---------------------')

print(100 * df.catalogs.value_counts()/len(df))

print('---------------------')
print("age:" ,df.age.isnull().value_counts()/len(df) * 100)

print('-------------------------------------')

print("gender:" ,df.gender.isnull().value_counts()/len(df) * 100)

print('-------------------------------------')

print("ownhome:" ,df.ownhome.isnull().value_counts()/len(df) * 100)

print('-------------------------------------')

print("married:" ,df.married.isnull().value_counts()/len(df) * 100)

print('-------------------------------------')

print("location:" ,df.location.isnull().value_counts()/len(df) * 100)

print('-------------------------------------')

print("salary:" ,df.salary.isnull().value_counts()/len(df) * 100)

print('-------------------------------------')

print("children:" ,df.children.isnull().value_counts()/len(df) * 100)

print('-------------------------------------')

print("history:" ,df.history.isnull().value_counts()/len(df) * 100)

print('-------------------------------------')

print("catalogs:" ,df.catalogs.isnull().value_counts()/len(df) * 100)

print('-------------------------------------')

print("amountspent:" ,df.amountspent.isnull().value_counts()/len(df) * 100)

print('-------------------------------------')
df[df.amountspent.isnull()]
df_drop_na = df.dropna()

df_drop_na = df_drop_na[df_drop_na.salary > 1].copy()

df_drop_na = df_drop_na[df_drop_na.amountspent != 0].copy()
len(df_drop_na)
df_drop_na.isnull().any()
sns.countplot(x='age', data = df_drop_na)

plt.suptitle('Frequency of observations by Age')
sns.countplot(x='gender', data = df_drop_na)

plt.suptitle('Frequency of observations by Gender')
sns.countplot(x='ownhome', data = df_drop_na)

plt.suptitle('Frequency of observations by Own Home')
sns.countplot(x='married', data = df_drop_na)

plt.suptitle('Frequency of observations by Married')
sns.countplot(x='location', data = df_drop_na)

plt.suptitle('Frequency of observations by Locations')
sns.countplot(x='children', data = df_drop_na)

plt.suptitle('Frequency of observations by Children')
sns.countplot(x='catalogs', data = df_drop_na)

plt.suptitle('Frequency of observations by Catalogs')
sns.countplot(x='history', data = df_drop_na)

plt.suptitle('Frequency of observations by History')
sns.distplot(df_drop_na['salary'])

plt.suptitle('Distribution of Salary')
sns.distplot(df_drop_na['amountspent'])

plt.suptitle('Distribution of Amount Spent')
sns.boxplot(x='salary', y = 'age', data=df_drop_na)

plt.suptitle('Salary levels by Age')
sns.boxplot(x='salary', y = 'gender', data=df_drop_na)

plt.suptitle('Salary levels by Gender')
sns.boxplot(x='salary', y = 'married', data=df_drop_na)

plt.suptitle('Salary levels by Married')
sns.boxplot(x='salary', y = 'ownhome', data=df_drop_na)

plt.suptitle('Salary levels by Own Home')
sns.boxplot(x='salary', y = 'location', data=df_drop_na)

plt.suptitle('Salary levels by Locations')
sns.boxplot(x='salary', y = 'history', data=df_drop_na)

plt.suptitle('Salary levels by History')
sns.boxplot(x='salary', y = 'catalogs', data=df_drop_na)

plt.suptitle('Salary levels by Catelogs')
sns.boxplot(x='amountspent', y = 'age', data=df_drop_na)

plt.suptitle('Amount Spent levels by Age')
sns.boxplot(x='amountspent', y = 'gender', data=df_drop_na)

plt.suptitle('Amount Spent levels by Gender')
sns.boxplot(x='amountspent', y = 'ownhome', data=df_drop_na)

plt.suptitle('Amount Spent levels by Own Home')
sns.boxplot(x='amountspent', y = 'location', data=df_drop_na)

plt.suptitle('Amount Spent levels by Location')
sns.boxplot(x='amountspent', y = 'children', data=df_drop_na)

plt.suptitle('Amount Spent levels by Children')
sns.boxplot(x='amountspent', y = 'history', data=df_drop_na)

plt.suptitle('Amount Spent levels by History')
sns.boxplot(x='amountspent', y = 'catalogs', data=df_drop_na)

plt.suptitle('Amount Spent levels by Catalogs')
sns.regplot(x='salary', y='amountspent', data=df_drop_na)
pd.DataFrame(round(100* df_drop_na.groupby('age')['gender'].value_counts()/len(df_drop_na),2))
pd.DataFrame(round(100* df_drop_na.groupby('age')['ownhome'].value_counts()/len(df_drop_na),2))
pd.DataFrame(round(100* df_drop_na.groupby('age')['married'].value_counts()/len(df_drop_na),2))
pd.DataFrame(round(100* df_drop_na.groupby('age')['location'].value_counts()/len(df_drop_na),2))
pd.DataFrame(round(100* df_drop_na.groupby('age')['children'].value_counts()/len(df_drop_na),2))
pd.DataFrame(round(100* df_drop_na.groupby('age')['history'].value_counts()/len(df_drop_na),2))
pd.DataFrame(round(100* df_drop_na.groupby('gender')['ownhome'].value_counts()/len(df_drop_na),2))
pd.DataFrame(round(100* df_drop_na.groupby('gender')['married'].value_counts()/len(df_drop_na),2))
pd.DataFrame(round(100* df_drop_na.groupby('gender')['location'].value_counts()/len(df_drop_na),2))
pd.DataFrame(round(100* df_drop_na.groupby('gender')['children'].value_counts()/len(df_drop_na),2))
pd.DataFrame(round(100* df_drop_na.groupby('gender')['history'].value_counts()/len(df_drop_na),2))
pd.DataFrame(round(100* df_drop_na.groupby('gender')['catalogs'].value_counts()/len(df_drop_na),2))
df_drop_na[['salary', 'amountspent', 'age']].groupby('age').describe().T
df_drop_na[['salary', 'amountspent', 'gender']].groupby('gender').describe().T
df_drop_na[['salary', 'amountspent', 'ownhome']].groupby('ownhome').describe().T
df_drop_na[['salary', 'amountspent', 'married']].groupby('married').describe().T
df_drop_na[['salary', 'amountspent', 'location']].groupby('location').describe().T
df_drop_na[['salary', 'amountspent', 'children']].groupby('children').describe().T
df_drop_na[['salary', 'amountspent', 'history']].groupby('history').describe().T
df_drop_na[['salary', 'amountspent', 'catalogs']].groupby('catalogs').describe().T
pd.get_dummies(df_drop_na).corr().style.background_gradient(cmap='coolwarm')