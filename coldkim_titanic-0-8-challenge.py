import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.head()
train_df.info()
test_df.info()
sns.countplot('Survived', data=train_df)
train_df.describe()
#문자열 데이터

train_df.describe(include=['O'])
pd.crosstab(train_df['Pclass'], train_df['Survived'], margins=True)
train_df[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean()
pd.crosstab(train_df['Sex'], train_df['Survived'], margins=True)
plt.xticks(np.arange(0,84,4))

plt.xlabel('Age')

plt.ylabel('Count')

plt.hist(train_df['Age'], bins=20)
g = sns.FacetGrid(train_df, col='Survived', height=7)

g.map(plt.hist, 'Age', bins=20)
train_df['Tmp']=""

plt.figure(figsize=(18,8))

plt.xticks(np.arange(0,110,5))

sns.violinplot('Age', 'Tmp', hue='Survived', data=train_df, scale='count', split=True).set_title('Age vs Survived')

train_df.drop('Tmp', axis=1, inplace=True)
cummulate_survival_ratio = []

for i in range(1,80):

    survival_ratio = train_df.loc[train_df['Age'] < i , 'Survived'].sum() / len(train_df.loc[train_df['Age'] < i, 'Survived'])

    cummulate_survival_ratio.append(survival_ratio)

plt.plot(cummulate_survival_ratio)
plt.figure(figsize=(18,8))

plt.yticks(np.arange(0,110,10))

sns.violinplot('Pclass', 'Age', hue='Survived', data=train_df, scale='count', split=True, inner='stick').set_title('Pclass and Age vs Survived')
plt.figure(figsize=(18,8))

plt.yticks(np.arange(0,110,10))

sns.violinplot('Sex', 'Age', hue='Survived', data=train_df, scale='count', split=True).set_title('Sex and Age vs Survived')
sns.pointplot('Pclass', 'Survived', hue='Sex', data=train_df, height=5)
pd.crosstab(train_df['Embarked'], train_df['Survived'], margins=True)
train_df[['Embarked', 'Survived']].groupby('Embarked', as_index=False).mean()
plt.xlabel('Fare')

plt.ylabel('Count')

plt.hist(train_df['Fare'], bins=40)
plt.xlabel('Fare')

plt.ylabel('Count')

plt.hist(test_df['Fare'], bins=40)
#test_df['Fare']에 null 값이 1개 존재하므로, 미리 최빈값으로 채워준다.

test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)



print(train_df['Fare'].describe())



combine = [train_df, test_df]

for dataset in combine:

    dataset['Fare'] = dataset['Fare'].apply(lambda x : 10 * np.log(x) if x>0 else 0)



print(train_df['Fare'].describe())



plt.xlabel('Fare')

plt.ylabel('Count')

plt.hist(train_df['Fare'], bins=40)
train_df['Tmp']=""

plt.figure(figsize=(18,8))

plt.xticks(np.arange(0,110,5))

sns.violinplot('Fare', 'Tmp', hue='Survived', data=train_df, scale='count', split=True).set_title('Fare vs Survived')

train_df.drop('Tmp', axis=1, inplace=True)
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



sns.countplot(train_df['FamilySize'])
sns.barplot('FamilySize', 'Survived', data=train_df, ci=None)
Embarked_mapping = {'C':0, 'Q':1, 'S':2}

for dataset in combine:

    dataset['Embarked'].fillna('S', inplace=True)

    dataset['Embarked'].replace({'C':0, 'Q':1, 'S':2}, inplace=True)

    dataset['Sex'].replace({'female':1, 'male':0}, inplace=True)

sns.heatmap(train_df[['Age', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize']].corr(), annot=True, square=True)
pc_values = [1,2,3]

fs_values = [1,2,3,4,5,6,7,8,11]

for dataset in combine:

    for i in pc_values:

        for j in fs_values:

            #Pclass와 FamilySize가 같은 그룹에서의 최빈값으로 채워줄거다

            age_guess = dataset.loc[(dataset['Pclass']==i)&(dataset['FamilySize']==j), 'Age'].dropna().median()

            dataset.loc[(dataset['Pclass']==i)&(dataset['FamilySize']==j&(dataset['Age'].isnull())), 'Age'] = age_guess

for dataset in combine:

    age = dataset['Age'].dropna().median()

    dataset['Age'].fillna(age, inplace=True)
print(train_df.info())

print(test_df.info())
plt.hist(train_df['Age'], bins=20)

plt.hist(test_df['Age'], bins=20)
cut_bins = [0,10,35,65,200]

cut_label = [3, 1, 2, 0] #생존율이 높은 구간이 3, 제일 낮은 구간이 0

for dataset in combine:

    dataset['AgeBand'] = pd.cut(train_df['Age'], bins=cut_bins, labels=cut_label, include_lowest=True)

    dataset['AgeToCat'] = dataset['AgeBand'].astype(int)
train_df[['AgeToCat', 'Survived']].groupby('AgeToCat', as_index=False).mean()
plt.hist(train_df['Fare'], bins=20)

plt.hist(test_df['Fare'], bins=20)
for dataset in combine:

    dataset['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby('FareBand', as_index=False).mean()
for dataset in combine:

    dataset['FareToCat'] = 0

    dataset.loc[(dataset['Fare'] > 20.682) & (dataset['Fare'] <= 26.71), 'FareToCat'] = 1

    dataset.loc[(dataset['Fare'] > 26.71) & (dataset['Fare'] <= 34.34), 'FareToCat'] = 2

    dataset.loc[ dataset['Fare'] > 34.34, 'FareToCat'] = 3

train_df.head()
train_df[['FareToCat', 'Survived']].groupby('FareToCat', as_index=False).mean()
for dataset in combine:

    dataset['Initial'] = dataset['Name'].str.extract(r'([A-Za-z]+)\.')
def color(x):

    c = 'red' if x ==0 else 'black'

    return 'color : {}'.format(c)

pd.crosstab(train_df['Sex'], train_df['Initial']).style.applymap(color)
pd.crosstab(test_df['Sex'], test_df['Initial']).style.applymap(color)
pd.crosstab(train_df['Survived'], train_df['Initial'])
for dataset in combine:

    dataset['InitialToCat'] = 5 #other initial

    dataset.loc[dataset['Initial'].isin(['Rev', 'Mr']), 'InitialToCat'] = 1 # Mr이 1

    dataset.loc[dataset['Initial'].isin(['Mlle', 'Mme', 'Ms', 'Miss']), 'InitialToCat'] = 2 # Miss가 2

    dataset.loc[dataset['Initial'].isin(['Mme', 'Mrs']), 'InitialToCat'] = 3 # Mrs가 3

    dataset.loc[dataset['Initial']=='Master', 'InitialToCat'] = 4 # Master가 4

pd.crosstab(train_df['Survived'], train_df['InitialToCat'])
train_df[['InitialToCat', 'Survived']].groupby('InitialToCat', as_index=False).mean()
#0: 5~ / 1: 1~2 / 2: 3~4

cut_bins = [1,3,5,100]

cut_label = [1, 2, 0] 

for dataset in combine:

    #dataset['AgeBand'] = pd.qcut(train_df['Age'], 5)

    #dataset['AgeBand'] = pd.cut(train_df['Age'], 5)

    dataset['FamilyToCat'] = pd.cut(train_df['FamilySize'], bins=cut_bins, labels=cut_label, include_lowest=True)

    dataset['FamilyToCat'] = dataset['FamilyToCat'].astype(int)
train_df[['FamilyToCat','Survived']].groupby('FamilyToCat', as_index=False).mean().sort_values(by='Survived')
train_df.info()
train_df.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'AgeBand', 'FareBand', 'Initial', 'FamilySize'], axis=1, inplace=True)

test_df.drop(['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'AgeBand', 'FareBand', 'Initial', 'FamilySize'], axis=1, inplace=True)
train_df.info()
test_df.info()
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

knn = KNeighborsClassifier(n_neighbors=8)

dt = DecisionTreeClassifier(max_depth=5, random_state=0)

vo = VotingClassifier(estimators=[('LR',lr), ('KNN',knn), ('DT', dt)], voting='soft')

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)



y_train_df = train_df['Survived']

x_train_df = train_df.drop('Survived', axis=1)

x_test_df = test_df.drop('PassengerId', axis=1)



scores = cross_val_score(vo, x_train_df, y_train_df, scoring='accuracy', cv=10)

print(np.mean(scores))

scores = cross_val_score(rf, x_train_df, y_train_df, scoring='accuracy', cv=10)

print(np.mean(scores))
dt.fit(x_train_df, y_train_df)

rf.fit(x_train_df, y_train_df)

importants = pd.Series(dt.feature_importances_, index=x_train_df.columns)

print(importants.sort_values(ascending=False))

importants = pd.Series(rf.feature_importances_, index=x_train_df.columns)

print(importants.sort_values(ascending=False))
def make_other_x(df):

    other_x = df.copy()

    other_x['FamilyAge'] = other_x['FamilyToCat'] * other_x['AgeToCat']

    return other_x

other_x = make_other_x(x_train_df)

other_x.head()
scores = cross_val_score(vo, other_x, y_train_df, scoring='accuracy', cv=10)

print(np.mean(scores))

scores = cross_val_score(rf, other_x, y_train_df, scoring='accuracy', cv=10)

print(np.mean(scores))



dt.fit(other_x, y_train_df)

rf.fit(other_x, y_train_df)

importants = pd.Series(dt.feature_importances_, index=other_x.columns)

print(importants.sort_values(ascending=False))

importants = pd.Series(rf.feature_importances_, index=other_x.columns)

print(importants.sort_values(ascending=False))
other_x_test = make_other_x(x_test_df)



rf.fit(other_x, y_train_df)

y_pred = rf.predict(other_x_test)

print(y_pred.shape)

y_pred
submit = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':y_pred})

print(submit.head())

submit.to_csv('submit.csv', index=False)