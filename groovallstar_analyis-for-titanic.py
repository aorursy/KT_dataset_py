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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



plt.style.use('seaborn')

sns.set(font_scale=2.5)

sns.set_style('darkgrid')



import missingno as msno



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
!ls ../input
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
df_train.head()
for col in df_train.columns:

    msg = 'column:{:>10}\t percent of NaN value:{:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)
df_train[col].isnull().sum()
f, ax = plt.subplots(1, 2, figsize=(10, 6))



df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('count plot - survived')

plt.show()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean()
df_train[['Pclass', 'Survived']].groupby(['Pclass']).sum()
df_train['Survived'].unique()
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True)
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
y_position = 1.02

f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#030303'], ax=ax[0])

ax[0].set_title('number of passengers', y=y_position)

ax[0].set_ylabel('count')

sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('pclass:survived vs dead', y=y_position)

plt.show()
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('survived vs sex')

sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('sex: survived vs dead')

plt.show()
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean()
pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)
sns.factorplot(x='Sex', y='Survived', hue='Pclass', data=df_train, saturation=.5, size=9, aspect=1)
df_train['Age'].max()
print('나이 제일 많은 탑승객:{:.1f} years'.format(df_train['Age'].max()))

print('제일 어린 탑승객:{:.1f} years'.format(df_train['Age'].min()))

print('탑승객 평균 나이:{:.1f} years'.format(df_train['Age'].mean()))
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
df_train[df_train['Survived'] == 1]['Age'].hist()
plt.figure(figsize=(8, 6))

df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution')

plt.legend(['1st class', '2nd class', '3rd class'])
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.kdeplot(df_train[(df_train['Survived'] == 0) & (df_train['Pclass'] == 1)]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived'] == 1) & (df_train['Pclass'] == 1)]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
change_age_range_survival_ratio = []



for i in range(1, 80):

    change_age_range_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))

    

plt.figure(figsize=(7, 7))

plt.plot(change_age_range_survival_ratio)

plt.title('survival rate change', y=1.02)

plt.ylabel('survival rate')

plt.xlabel('range of age')

plt.show()
i = 10

df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived'])
f, ax = plt.subplots(1, 2, figsize=(18, 8))

sns.violinplot('Pclass', 'Age', hue='Survived', data=df_train, scale='count', split=True, ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0, 110, 10))



sns.violinplot('Sex', 'Age', hue='Survived', data=df_train, scale='count', split=True, ax=ax[1])

ax[1].set_title('sex and age vs survived')

ax[1].set_yticks(range(0, 110, 10))



plt.show()
f, ax = plt.subplots(1, 1, figsize=(7, 7))

df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived')
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_index
f, ax = plt.subplots(2, 2, figsize=(15, 12))

sns.countplot('Embarked', data=df_train, ax=ax[0, 0])

ax[0, 0].set_title('(1) No. Of Passengers Boared')



sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax[0, 1])

ax[0, 1].set_title('(2) No. male-female split for embarked')



sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1, 0])

ax[1, 0].set_title('(3) Embarked vs Survived')



sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1, 1])

ax[1, 1].set_title('(4) Embarked vs Pclass')



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()

df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
print('Maximum size of Family', df_train['FamilySize'].max())

print('Minimum size of Family', df_train['FamilySize'].min())
f, ax = plt.subplots(1, 3, figsize=(40, 10))

sns.countplot('FamilySize', data=df_train, ax=ax[0])

ax[0].set_title('(1) No. Of passenger boarded')



sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('(2) No. Survived countplot depending on FamilySize')



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize')



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
fig, ax = plt.subplots(1,1, figsize=(8, 8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness:{:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_train['Fare']
df_train['Age'].isnull().sum()
df_train['Initial'] = df_train['Name'].str.extract('([A-Za-z]+)\.')

df_test['Initial'] = df_test['Name'].str.extract('([A-Za-z]+)\.')
df_train.head()
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')
df_train['Initial'].replace(

            ['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

            ['Miss','Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mr'], inplace=True)



df_test['Initial'].replace(

            ['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

            ['Miss','Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mr'], inplace=True)
df_train.groupby('Initial')['Survived'].mean().plot.bar()
df_all = pd.concat([df_train, df_test])
df_all.reset_index(drop=True)
df_all.groupby('Initial').mean()
df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial'] == 'Mr'), 'Age'] = 33

df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial'] == 'Mr'), 'Age']
df_train.loc[(df_train['Initial'] == 'Mr'), 'Age']
df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial'] == 'Mr'), 'Age'] = 33

df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial'] == 'Mrs'), 'Age'] = 37

df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial'] == 'Master'), 'Age'] = 5

df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial'] == 'Miss'), 'Age'] = 22

df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial'] == 'Other'), 'Age'] = 45



df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial'] == 'Mr'), 'Age'] = 33

df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial'] == 'Mrs'), 'Age'] = 37

df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial'] == 'Master'), 'Age'] = 5

df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial'] == 'Miss'), 'Age'] = 22

df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial'] == 'Other'), 'Age'] = 45
df_train['Age'].isnull().sum()
df_train['Embarked'].isnull().sum()
df_train['Embarked'].fillna('S', inplace=True)
df_train['Age_cat'] =0

df_test['Age_cat'] = 0
df_train.head()
def category_age(x):

    if x < 10:

        return 0

    elif x < 20:

        return 1

    elif x < 30:

        return 2

    elif x < 40:

        return 3

    elif x < 50:

        return 4

    elif x < 60:

        return 5

    elif x < 70:

        return 6

    else:

        return 7
df_train['Age_cat'] = df_train['Age'].apply(category_age)

df_test['Age_cat'] = df_test['Age'].apply(category_age)
df_train.drop(['Age'], axis=1, inplace=True)

df_test.drop(['Age'], axis=1, inplace=True)
df_train['Initial'] = df_train['Initial'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4})

df_test['Initial'] = df_test['Initial'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4})
df_test.head()
df_train['Embarked'] = df_train['Embarked'].map({'C':0, 'Q':1, 'S':2})

df_test['Embarked'] = df_test['Embarked'].map({'C':0, 'Q':1, 'S':2})
df_train.Embarked.isnull().sum()
df_train['Sex'].unique()
df_train['Sex'] = df_train['Sex'].map({'female':0, 'male':1})

df_test['Sex'] = df_test['Sex'].map({'female':0, 'male':1})
heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']]
heatmap_data.corr()
colormap = plt.cm.BuGn

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={'size':16}, fmt='.2f')
df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_train.head()
df_test.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split
X_train = df_train.drop('Survived', axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)
model = RandomForestClassifier()

model.fit(X_tr, y_tr)
prediction = model.predict(X_vld)
print('총 {}명 중 {:.2f}% 정확도로 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
(prediction == y_vld).sum() / prediction.shape[0]
from pandas import Series
feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature Importance')

plt.ylabel('Feature')

plt.show()
X_test
submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission.head()
prediction = model.predict(X_test)

submission['Survived'] = prediction
submission.to_csv('./my_first_submission.csv', index=False)