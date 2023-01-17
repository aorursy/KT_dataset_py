import pandas as pd # 데이터 처리 라이브러리

import numpy as np # 수학 연산 라이브러리

import matplotlib.pyplot as plt # 데이터 시각화 

import seaborn as sns # 데이터 시각화
path = '../input/'

train_df = pd.read_csv(path+'train.csv')

test_df = pd.read_csv(path+'test.csv')
print("트레이닝 데이터 개수 : ", train_df.shape)

print("테스트 데이터 개수 : ", test_df.shape)
train_df.columns
train_df.head(5)
train_df.isnull()
train_df.isnull().sum(axis = 0)
na_df = train_df.isnull().sum(axis = 0)

na_df.reset_index()
na_df = na_df.reset_index()

na_df.columns = ['feature', 'count']

na_df[na_df['count'] > 0]
na_df = test_df.isnull().sum().reset_index()

na_df.columns = ['feature', 'count']

na_df[na_df['count'] > 0]
train_df.describe()
train_df['Survived'].describe(percentiles = [.61, .62])
train_df['Parch'].describe(percentiles = [.75, .80])
train_df['SibSp'].describe(percentiles = [.68, .69])
train_df['Age'].describe(percentiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .99])
pd.value_counts(train_df['SibSp']) # 해당 승객의 같이 탑승한 형제,자매, 배우자 숫자
pd.value_counts(train_df['Parch']) # 해당 승객의 같이 탑승한 부모님, 아이 숫자
train_df.describe(include = ['O'])
male_df = train_df[ train_df['Sex'] == 'male']

male_df['Survived'].mean()
female_df = train_df[ train_df['Sex'] == 'female']

female_df['Survived'].mean()
group_df = train_df.groupby(by = 'Survived')

group_df
group_df.agg('mean')
group_df.agg('sum')
group_df.agg('max')
group_df = train_df[['Pclass', 'Survived']].groupby(['Pclass']).agg('mean')

group_df.sort_values(by = 'Survived',ascending = False)
group_df = train_df[['Sex', 'Survived']].groupby(by = ['Sex']).agg('mean')

group_df.sort_values(by = 'Survived', ascending = False)
group_df = train_df[['SibSp', 'Survived']].groupby(by = ['SibSp']).agg('mean')

group_df.sort_values(by = 'Survived', ascending = False)
group_df = train_df[['Parch', 'Survived']].groupby(by = ['Parch']).agg('mean')

group_df.sort_values(by = 'Survived', ascending = False)
# FacetGrid

g = sns.FacetGrid(data = train_df, col = 'Survived')

g.map(plt.hist, 'Age', bins = 20)
grid = sns.FacetGrid(data = train_df, row = 'Pclass', col = 'Survived', size = 2.2, aspect = 1.6)

grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)

grid.add_legend()
grid = sns.FacetGrid(train_df, row = 'Embarked', size = 2.2, aspect = 1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = 'deep')

grid.add_legend()
grid = sns.FacetGrid(data = train_df, row = 'Embarked', col = 'Survived', size = 2.2, aspect = 1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha = 0.5, ci = 95)

grid.add_legend()