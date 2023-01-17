import numpy as np

import pandas as pd



# The following two modules matplotlib and seaborn are for plots

import matplotlib.pyplot as plt

import seaborn as sns # Comment this if seaborn is not installed

%matplotlib inline



# The module re is for regular expressions

import re
path = '../input/titanic/'

df = pd.read_csv(path + 'train.csv')
df
df.shape
df.head()
df.columns
df[df['Sex'] == "female"].head()
df.loc[df['Age']>70, ['Name', 'Survived']]
df.iloc[100:106]
plt.axis('equal')

plt.pie(df['Survived'].value_counts(), labels=('Died', "Survived"));
sns.barplot(x = 'Sex', y = 'Survived', data = df);
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df);
sns.pointplot(x='Sex', y='Survived', hue='Pclass', data=df);
df.isnull().sum()
df.head()
df.loc[:20, 'Name'].values
re.findall("\w\w[.]", 'Braund, Mr. Owen Harris')
re.findall("\w\w[.]", 'Braund, Mr. Owen Harris')[0]
re.findall("\w\w[.]", 'Heikkinen, Miss. Laina')[0]
# Fill in below:

re.findall("FILL IN HERE", 'Heikkinen, Miss. Laina')[0]
get_title('Futrelle, Mrs. Jacques Heath (Lily May Peel)')
get_title('Simonius-Blumer, Col. Oberst Alfons')
df.head()
df.groupby('Title')
df.groupby('Title').median()
df.groupby('Title').mean()
df['MedianAge'] = df.groupby('Title')['Age'].transform("median")

df.head(15)
df['Age'] = df['Age'].fillna(df['MedianAge'])

df.head()
df = df.drop('MedianAge', axis=1)

df.isnull().sum()
df = df.replace({'male': 0, 'female': 1})

df.dtypes
pd.get_dummies(df['Embarked']).head()
port_df = pd.get_dummies(df['Embarked'], prefix='Port')

port_df.head()
df = pd.concat([df, port_df], axis=1)

df.head()
df.corr()
correlation_matrix = df.corr();

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(correlation_matrix);
df1 = pd.DataFrame({'CourseCode': ['PHYS024', 'CSCI35', 'ENGR156'], 

                   'CourseName': ['Mechanics and Wave Motion', 

                                  'Computer Science for Insight',

                                 'Intro to Comm & Info Theory']})



df2 = pd.DataFrame({'Professor': ['Zachary Dodds', 'Vatche Sahakian', 

                                  'Timothy Tsai', 'Brian Shuve'],

                    'CourseCode': ['CSCI35', 'PHYS024',  'ENGR156', 'PHYS024']})



df1.head()
df2.head()
pd.merge(df2, df1)
path = '../input/instacart-market-basket-analysis/'

dfa = pd.read_csv(path + 'aisles.csv')

dfd = pd.read_csv(path + 'departments.csv')

dfp = pd.read_csv(path + 'products.csv')

dfo = pd.read_csv(path + 'order_products__train.csv')
df
dfp.head()
df
df
df
df
df.shape
sns.barplot(x='Family', y='Survived', data=df);
df['Ticket'].value_counts()[:15]
df.head()
df[df['GroupSize'] != df['Family']].shape[0], df[df['GroupSize'] != df['TicketCount']].shape[0]