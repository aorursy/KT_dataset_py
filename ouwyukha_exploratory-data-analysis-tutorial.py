import pandas as pd

import math

from matplotlib import pyplot as plt

import seaborn as sns
pd.set_option("display.max_columns", 100) # set max columns when displaying pandas DataFrame

pd.set_option("display.max_rows", 200) # set max rows when displaying pandas DataFrame



plt.rcParams["figure.figsize"] = (10,6) # define figure size of pyplot
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.shape
len(df)
df
df.head(5)
df.sort_values(by=['Age'], ascending=False, inplace=False)
df.describe()
df.info()
df.index
df.columns
df.columns.tolist()
df.nunique()
df['Name']
df['Name'].values[:5]
df[['Name']].head(5)
df[['Name', 'Sex', 'Embarked', 'Survived']].head(5)
df[1:3]
df.iloc[:5]
df = df.iloc[:, 1:]

df
df.iloc[-3:, -4:]
alist = [4,8,6]

df.iloc[alist]
df.loc[[10, 20, 30]]
df.loc[:3, ['Name', 'Sex', 'Age']]
df[:10].loc[:, ['Survived', 'Name', 'Age']].iloc[2:5, 1:].sample(3)
df['Age'].isnull() # or .isna()
df[df['Age'].isnull()].head(3)
df[~df['Age'].isnull()].head(3) # or simply use .notnull() or notna()
df[df['Name'].str.contains('Master.')].sample(5)
df[(df['Age'].isnull()) & (df['Sex']=='male')].head(3)
unique_cabin = df['Cabin'].unique()

unique_cabin
A_cabin = []

for cabin in unique_cabin:

    if isinstance(cabin, float): # Skip Null Value

        continue

        

    if 'A' in cabin:

        A_cabin.append(cabin)

A_cabin
df[df['Cabin'].isin(A_cabin)]
null_age = df[df['Age'].isnull()].copy() # Use copy() to create a copy of DataFrame to prevent modification leakage.
null_age.groupby('Pclass').size()
null_age.groupby('Pclass').size() / len(null_age) * 100
df.groupby('Pclass')['Fare'].mean()
df.groupby('Pclass')['Fare'].mean().reset_index().rename(columns={'Fare':'Mean_Fare'})
df.groupby('Pclass')['Fare'].mean().to_frame(name='Mean_Fare')
df[['SibSp', 'Parch']].sum(axis=0)
df[['SibSp', 'Parch']].sum(axis=1)
df.groupby(['Pclass', 'Sex'])['Survived'].sum().to_frame(name='Survived')
pe_group = df.groupby(['Pclass', 'Embarked'])
class_survivability = pe_group['Survived'].apply(lambda x: x.sum() / x.count()*100).reset_index()

class_survivability
sns.barplot(x = "Pclass", y="Survived", hue="Embarked", data=class_survivability)
df.groupby('Embarked')['Fare'].agg(['min', 'max', 'mean', 'median', 'sum', 'count'])
df['TEST'] = 1

df.head(3)
df['TEST'] = df['Age'] ** 2

df[['Age', 'TEST']].head(3)
df.drop(columns=['TEST'], inplace=True)

df.head(3)
df1 = df[['Survived']].copy()

df2 = df[['Pclass', 'Name']].copy()



print(df1.shape, df2.shape)
pd.concat([df1, df2], axis=1)
df1 = df.iloc[[1,2,3]].copy()

df2 = df.iloc[[11,12,13]].copy()

df3 = df.iloc[[23,33,43]].copy()



print(df1.shape, df2.shape, df3.shape)
pd.concat([df1, df2, df3], axis=0)
avg_fare = df.groupby('Pclass')['Fare'].mean().reset_index().rename(columns={'Fare':'AVG_Fare'})

avg_fare
df = pd.merge(df, avg_fare, how='left', on=['Pclass'])

df.head(5)
df.isnull().sum()
male_filler_age = df.loc[df['Sex']=='male', 'Age'].median()

female_filler_age = df.loc[df['Sex']=='female', 'Age'].median()



print('Male Filler Age:', male_filler_age)

print('Female Filler Age:', female_filler_age)
df.loc[(df['Age'].isnull()) & (df['Sex']=='male'), 'Age'] = male_filler_age

df.loc[(df['Age'].isnull()) & (df['Sex']=='female'), 'Age'] = female_filler_age
df[df['Age'].isnull()]
df['Cabin'].head(5)
df['Cabin'].fillna('Unknown', inplace=True)
df['Cabin'].head(5)
df[df['Embarked'].isnull()]
# subset, columns or rows that will be considered with default value of None (consider all of them).

df.dropna(subset=['Embarked'], inplace=True)
df.isnull().sum()
df['Fare'].count()
df['Fare'].max()
df['Fare'].min()
df['Name'].max()
df['Name'].min()
df['Ticket'].mode()
df['Ticket'].value_counts(dropna=False) # Use dropna=False to count the frequency of NaN value too
VC_Ticket = df['Ticket'].value_counts()



VC_Ticket[VC_Ticket==VC_Ticket.iloc[0]] 
df['Age'].mean()
df['Age'].median()
Q1 = df['Age'].quantile(q=0.25)

Q1
Q2 = df['Age'].quantile(q=0.5) # or usually called as median

Q2
Q3 = df['Age'].quantile(q=0.75)

Q3
IQR = df.quantile(0.75) - df.quantile(0.25)

IQR
df.var()
df.std()
df.skew()
df['Fare'].plot(kind='hist')
df.kurtosis()
df['SibSp'].plot(kind='hist')
df.cov()
df.corr(method='pearson')
sns.heatmap(df.corr(), annot=True)