import pandas as pd # 데이터 처리 라이브러리

import numpy as np # 수학 연산 라이브러리

import matplotlib.pyplot as plt # 데이터 시각화 

import seaborn as sns # 데이터 시각화



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
path = '../input/'

train_df = pd.read_csv(path+'train.csv')

test_df = pd.read_csv(path+'test.csv')
print("트레이닝 데이터 개수 : ", train_df.shape)

print("테스트 데이터 개수 : ", test_df.shape)
train_df.head()
columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Survived"]

train_df_simple = train_df[columns]

train_df_simple.head()
mapping = {"male" : 0, "female" : 1}

train_df_simple["Sex"] = train_df_simple["Sex"].replace(mapping)

train_df_simple.head()
train_df_simple = train_df_simple.fillna(-1)
# 데이터 구성

X_train = train_df_simple.iloc[:500, 0:5].values

y_train = train_df_simple.iloc[:500, 5].values



X_val = train_df_simple.iloc[500:, 0:5].values

y_val = train_df_simple.iloc[500:, 5].values



# 모델 만들기

model = LogisticRegression()



# 모델 학습하기

model = model.fit(X_train, y_train)



# 모델 성능 평가하기

y_pred = model.predict(X_val)

print(accuracy_score(y_val, y_pred))
train_df.head()
train_df['Cabin_is_value'] = (train_df['Cabin'].isnull() == False).astype('int')

test_df['Cabin_is_value'] = (test_df['Cabin'].isnull() == False).astype('int')
train_df = train_df.drop(labels = ['Ticket', 'Cabin'], axis = 1)

test_df = test_df.drop(labels = ['Ticket', 'Cabin'], axis = 1)

combine = [train_df, test_df]
train_df.head()
train_df.Name
train_df.Name.str.extract(pat = '([A-Za-z])')
train_df.Name.str.extract(pat = '( [A-Za-z])')
train_df.Name.str.extract(pat = '([A-Za-z]+)')
train_df.Name.str.extract(pat = '( [A-Za-z]+)\.')
train_df.Name.str.extract(pat = '( [A-Za-z]+)\.', expand = False)
train_df.Name.str.extract(pat = '( [A-Za-z]+)\.').isnull().sum()
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

pd.crosstab(train_df['Title'], train_df['Sex'])
train_df[['Title', 'Survived']].groupby(by = 'Title').agg('mean').sort_values(by = 'Survived', ascending = False)
rare = ['Lady','Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(to_replace = rare, value = 'Rare')

    dataset['Title'] = dataset['Title'].replace(to_replace = 'Mlle', value = 'Miss')

    dataset['Title'] = dataset['Title'].replace(to_replace = 'Ms', value = 'Miss')

    dataset['Title'] = dataset['Title'].replace(to_replace = 'Mme', value = 'Mrs')



train_df[['Title', 'Survived']].groupby(by = 'Title', as_index = False).agg('mean').sort_values(by = 'Survived', ascending = False)
title_mapping = {'Mr':1, 'Rare' : 2, 'Master' : 3, 'Miss' : 4, 'Mrs' : 5}

for dataset in combine:

    dataset['Title'] = dataset['Title']. replace(title_mapping)



train_df.head()
train_df = train_df.drop(labels = ['Name'], axis = 1)

test_df = test_df.drop(labels = ['Name'], axis = 1)

combine = [train_df, test_df]
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].replace({'male' : 0, 'female' : 1}).astype(int)
train_df.head()
train_df['Embarked'].isnull().sum()
test_df['Embarked'].isnull().sum()
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train_df[['Embarked', 'Survived']].groupby(by = 'Embarked', as_index = False).agg('mean').sort_values(by = 'Survived', ascending = False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].replace({'S':0,'Q':1,'C':2}).astype('int')
train_df.head()
print("The number of missing values of Age in train_df :", train_df['Age'].isnull().sum())

print("The number of missing values of Age in test_df :", test_df['Age'].isnull().sum())
grid = sns.FacetGrid(train_df, row = 'Pclass', col = 'Sex', size = 2.2, aspect = 1.6)

grid.map(plt.hist, 'Age', alpha = .5, bins = 20)

grid.add_legend()
guess_ages = np.zeros(shape = (3, 2))

guess_ages
guess_ages = []

Pclass_values = [1, 2, 3]

Sex_values = [0, 1]



for pclass in Pclass_values:

    temp = []

    for sex in Sex_values:

        data = train_df[(train_df['Pclass'] == pclass) & (train_df['Sex'] == sex)]

        median = data['Age'].median()

        temp.append(median)

    guess_ages.append(temp)



guess_ages
train_df[train_df['Age'].isnull()]
train_df[train_df['Age'].isnull()].index
train_df.loc[5,'Age'] = 25.0
train_df.head(6)
for na_idx in train_df[train_df['Age'].isnull()].index:

    pclass = train_df.loc[na_idx, 'Pclass']

    sex = train_df.loc[na_idx, 'Sex']

    train_df.loc[na_idx, 'Age'] = guess_ages[pclass - 1][sex]
train_df['Age'].isnull().sum()
for na_idx in test_df[test_df['Age'].isnull()].index:

    pclass = test_df.loc[na_idx, 'Pclass']

    sex = test_df.loc[na_idx, 'Sex']

    test_df.loc[na_idx, 'Age'] = guess_ages[pclass - 1][sex]
test_df['Age'].isnull().sum()
train_df['Age'] = train_df['Age'].astype('int')

test_df['Age'] = test_df['Age'].astype('int')
pd.cut(train_df['Age'], 5)
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

band_df = train_df[['AgeBand', 'Survived']].groupby(by = 'AgeBand', as_index = False).agg('mean').sort_values(by = 'Survived', ascending = False)

band_df
dic = {}

for a,b in zip(band_df['AgeBand'], range(0,5)):

    dic[a] = b

dic
train_df['AgeBand'].replace(dic)
train_df['AgeBand'] = train_df['AgeBand'].replace(dic)
test_df['Age']
def find_band(x, dic):

    for interval in dic.keys():

        if x in interval:

            return dic[interval]    
test_df['Age'].map(lambda x: find_band(x, dic))
test_df['AgeBand'] = test_df['Age'].map(lambda x: find_band(x, dic))
train_df.head()
print("The number of missing values of Fare in train data : ", train_df['Fare'].isnull().sum())

print("The number of missing values of Fare in test data : ", test_df['Fare'].isnull().sum())
median = train_df['Fare'].median()

test_df['Fare'] = test_df['Fare'].fillna(median)
print("The number of missing values of Fare in test data : ", test_df['Fare'].isnull().sum())
#실습

train_df['FareBand'] = pd.cut(train_df['Fare'], bins = 4)

band_df = train_df[['FareBand', 'Survived']].groupby(by = 'FareBand', as_index = False).agg('mean').sort_values(by = 'Survived', ascending = False)

band_df
dic = {}

for a,b in zip(band_df['FareBand'], range(0,4)):

    dic[a] = b

dic
train_df['FareBand'] = train_df['FareBand'].replace(dic)
test_df['FareBand'] = test_df['Fare'].map(lambda x : find_band(x, dic))
X = train_df.drop(labels=['PassengerId','Survived'], axis = 1).values

y = train_df['Survived'].values



X_train = X[:500, :]

y_train = y[:500]

X_val = X[500:, :]

y_val = y[500:]

# X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.3, random_state = 1)



# X_test = test_df.drop(labels = ['PassengerId'], axis = 1).values
model = RandomForestClassifier(n_estimators = 100)

model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print(accuracy_score(y_val, y_pred))