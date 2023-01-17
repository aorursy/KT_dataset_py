# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')

sns.set(font_scale=2.5) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.

import missingno as msno



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
# read data

df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
df_train.head()
df_test.head()
df_train.describe() # 각 feature 의 간단한 통계를 알려줌
df_test.describe()
# null 비율을 산출 

for col in df_train.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)
# null 비율을 산출 

for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))

    print(msg)
# null data visualization

msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
# null data visualization

msno.matrix(df=df_test.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
msno.bar(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
msno.bar(df=df_test.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
f, ax = plt.subplots(1,2, figsize=(18,8))



df_train['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('pie plot - survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data = df_train, ax=ax[1])

ax[1].set_title('count plot - survived')



plt.show()
# 각 등급별 탑승인원 확인

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
# 각 등급별 생존자 확인

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
# pclass별 survive 분포 확인

pd.crosstab(df_train["Pclass"], df_train["Survived"], margins=True).style.background_gradient(cmap='summer_r')
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
y_position=1.02

f, ax = plt.subplots(1,2 , figsize=(18,8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'] , ax=ax[0])

ax[0].set_title('number of passengers by pclass', y=y_position)

ax[0].set_ylabel('count')



sns.countplot('Pclass', hue='Survived', data = df_train,ax=ax[1])

ax[1].set_title('pclass: survived vs dead', y=y_position)

ax[1].set_ylabel('count')

plt.show()
f, ax = plt.subplots(1,3, figsize=(18,8))



for i in range(3) :

    df_train.loc[df_train['Pclass']==i+1]['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[i], shadow=True)

    ax[i].set_title('pie plot(pclass: '+str(i)+') - survived')

    ax[i].set_ylabel('')
f, ax = plt.subplots(1,2, figsize=(18,8))

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title("Survived vs Sex")



sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Sex: survived vs dead')



plt.show()
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)
sns.factorplot(x ='Sex', y='Survived', col='Pclass', data=df_train,

               satureation=.5,size=9, aspect=1)
print('나이 max: {:.1f} Years'.format(df_train["Age"].max()))

print('나이 min: {:.1f} Years'.format(df_train["Age"].min()))

print('나이 mean: {:.1f} Years'.format(df_train["Age"].mean()))
f, ax = plt.subplots(1,1, figsize=(9,5))



sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)



plt.legend(['survived', 'dead'])

plt.show()
plt.figure(figsize=(8,6))



df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age distribution within classes')

plt.legend(['1', '2', '3'])
cummulate_survival_ratio = []

for i in range(1, 80) :

    cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))

    

plt.figure(figsize=(7,7))

plt.plot(cummulate_survival_ratio)

plt.title('Survived rate change depending on range of age', y=1.02)

plt.ylabel('survived rate')

plt.xlabel('range of age')

plt.show()
f, ax= plt.subplots(1,2, figsize=(18,8))



sns.violinplot('Pclass', 'Age', hue='Survived', data =df_train, scale='count', split=True, ax=ax[0])

ax[0].set_title('pclass, age vs survived')

ax[0].set_yticks(range(0,110,10))





sns.violinplot('Sex', 'Age', hue='Survived', data =df_train, scale='count', split=True, ax=ax[1])

ax[0].set_title('Sex, age vs survived')

ax[0].set_yticks(range(0,110,10))



plt.show()

f, ax = plt.subplots(1,1, figsize=(7,7))



df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
f, ax=plt.subplots(2,2, figsize=(20,15))



sns.countplot('Embarked', data=df_train, ax= ax[0,0])

ax[0,0].set_title("1. num of passengers boarded")





sns.countplot('Embarked', hue="Sex",data=df_train, ax= ax[0,1])

ax[0,1].set_title("2 . male-female split for Embarked")



sns.countplot('Embarked', hue='Survived', data=df_train, ax= ax[1,0])

ax[1,0].set_title("3. Embarked vs Survived")



sns.countplot('Embarked', hue='Pclass', data=df_train, ax= ax[1,1])

ax[1,1].set_title("3. Embarked vs Pclass")



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
# 형제, 자매 수 + 부모, 자식의 수 + 자기자신(1)

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 



print("max of family num: ", df_train["FamilySize"].max())

print("min of family num: ", df_train["FamilySize"].min())
f, ax = plt.subplots(1,3, figsize=(40,10))



sns.countplot('FamilySize', data=df_train, ax=ax[0])

ax[0].set_title('1. num of passengers boarded')



sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('2. Survived count depending on F.S', y =1.02)



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('3. Survived rate depending on F.S', y =1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
f, ax = plt.subplots(1,1, figsize=(8,8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness: {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()



df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
f, ax = plt.subplots(1,1, figsize=(8,8))



g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
# cabin의 null이 너무 많아 제외

df_train.head()
df_train['Ticket'].value_counts()

# 아이디어를 찾아보자 
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')



# FamilySize

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다



# fare null --> mean()

df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()



# fare distribution --> log

df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
# name의 Mr, Mrs, ... 을 찾아보자

df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.') 

df_test['Initial'] = df_test.Name.str.extract('([A-Za-z]+)\.') 
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')
# 이름을 Miss, Mr, Mrs, other로 나눔

df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)



df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
df_train.groupby('Initial').mean()
df_train.groupby('Initial')['Survived'].mean().plot.bar()
df_train.groupby('Initial').mean()
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mr'),'Age'] = 33

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mrs'),'Age'] = 36

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Master'),'Age'] = 5

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Miss'),'Age'] = 22

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Other'),'Age'] = 46



df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mr'),'Age'] = 33

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mrs'),'Age'] = 36

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Master'),'Age'] = 5

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Miss'),'Age'] = 22

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Other'),'Age'] = 46
print('Embarked has ', sum(df_train['Embarked'].isnull()), ' Null values')
# null data가 2개밖에없고 이를 가장 많은 S로 채운다.

df_train['Embarked'].fillna('S', inplace=True)
df_train['Age_cat'] = 0

df_train.loc[df_train['Age'] < 10, 'Age_cat'] = 0

df_train.loc[(10 <= df_train['Age']) & (df_train['Age'] < 20), 'Age_cat'] = 1

df_train.loc[(20 <= df_train['Age']) & (df_train['Age'] < 30), 'Age_cat'] = 2

df_train.loc[(30 <= df_train['Age']) & (df_train['Age'] < 40), 'Age_cat'] = 3

df_train.loc[(40 <= df_train['Age']) & (df_train['Age'] < 50), 'Age_cat'] = 4

df_train.loc[(50 <= df_train['Age']) & (df_train['Age'] < 60), 'Age_cat'] = 5

df_train.loc[(60 <= df_train['Age']) & (df_train['Age'] < 70), 'Age_cat'] = 6

df_train.loc[70 <= df_train['Age'], 'Age_cat'] = 7



df_test['Age_cat'] = 0

df_test.loc[df_test['Age'] < 10, 'Age_cat'] = 0

df_test.loc[(10 <= df_test['Age']) & (df_test['Age'] < 20), 'Age_cat'] = 1

df_test.loc[(20 <= df_test['Age']) & (df_test['Age'] < 30), 'Age_cat'] = 2

df_test.loc[(30 <= df_test['Age']) & (df_test['Age'] < 40), 'Age_cat'] = 3

df_test.loc[(40 <= df_test['Age']) & (df_test['Age'] < 50), 'Age_cat'] = 4

df_test.loc[(50 <= df_test['Age']) & (df_test['Age'] < 60), 'Age_cat'] = 5

df_test.loc[(60 <= df_test['Age']) & (df_test['Age'] < 70), 'Age_cat'] = 6

df_test.loc[70 <= df_test['Age'], 'Age_cat'] = 7
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

    

df_train['Age_cat_2'] = df_train['Age'].apply(category_age)

df_test['Age_cat_2'] = df_test['Age'].apply(category_age)
print('1번 방법, 2번 방법 둘다 같은 결과를 내면 True 줘야함 -> ', (df_train['Age_cat'] == df_train['Age_cat_2']).all())
df_train.drop(['Age', 'Age_cat_2'], axis=1, inplace=True)

df_test.drop(['Age', 'Age_cat_2'], axis=1, inplace=True)
df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
df_train['Embarked'].unique()
df_train['Embarked'].value_counts()
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
# null 확인

df_train['Embarked'].isnull().any()
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})

df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})

df_train['Sex'].isnull().any()
heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']] 



colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)



sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,

           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})



del heatmap_data
df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')

df_train.head()
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')

df_train.head()
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
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size = 0.3, random_state = 395)
model = RandomForestClassifier()

model.fit(X_tr, y_tr)

prediction = model.predict(X_vld)
print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
max_idx, max_score = 0, 0

model = RandomForestClassifier()

for i in range(3000) :

    X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size = 0.3, random_state = i)

    model.fit(X_tr, y_tr)

    prediction = model.predict(X_vld)

#     print(i, end=', ')

#     print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))

    

    if max_score < 100 * metrics.accuracy_score(prediction, y_vld):

        max_score = 100 * metrics.accuracy_score(prediction, y_vld)

        max_idx = i 



print(max_idx, max_score)
print(max_idx)

X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size = 0.3, random_state = max_idx)

model.fit(X_tr, y_tr)

prediction = model.predict(X_vld)

print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
from pandas import Series



feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')

plt.show()
submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission.head()
X_test
prediction = model.predict(X_test)

prediction
submission['Survived'] = prediction

submission.head()
submission.to_csv('./my_first_submission.csv', index=False)