# load dependencies for data analysis, wrangling and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# load the data
train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
df = pd.concat([train_set, test_set])
train_set.head()
df.info()
train_set.describe()
train_set.hist(figsize=(12,12))
train_set.describe(include=['O'])
print(train_set.Survived.count())
print(train_set.Survived.groupby(train_set.Survived).describe())
print('survived: %.2f%%' % ((342 / 891) * 100))
train_set.Survived.groupby(train_set.Survived)
print('training set includes %.2f%% of the 2224 passengers' % ((891 / 2224) * 100))
print(df.isnull().any())
print(df.isnull().sum())
train_set.corr()
print(train_set.Pclass.groupby(train_set.Pclass).describe())
train_set[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False)
train_set[['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)
train_set[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)
train_set.Sex = train_set.Sex.astype('category')
train_set.Sex = train_set.Sex.cat.codes
train_set.head()
sns.pairplot(train_set[['Survived', 'Sex', 'Fare', 'Pclass', 'SibSp']], kind='reg')
plt.show()
