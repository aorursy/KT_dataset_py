import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import keras

import sklearn



plt.style.use('seaborn')

sns.set(font_scale=2.5) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.

import missingno as msno



#ignore warnings

import warnings

warnings.filterwarnings('ignore') # 워닝 메세지를 생략해 줍니다. 차후 버전관리를 위해 필요한 정보라고 생각하시면 주석처리 하시면 됩니다.



%matplotlib inline
os.listdir("../input/2019-1st-ml-month-with-kakr")
df_train = pd.read_csv('../input/2019-1st-ml-month-with-kakr/train.csv')

df_test = pd.read_csv('../input/2019-1st-ml-month-with-kakr/test.csv')

df_submit = pd.read_csv('../input/2019-1st-ml-month-with-kakr/sample_submission.csv')
df_train.shape, df_test.shape, df_submit.shape
df_train.columns
df_train.head()
df_test.head()
df_submit.head()
f, ax = plt.subplots(1, 2, figsize=(18, 8))



df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived')



plt.show()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().plot.bar()
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, 

               size=6, aspect=1.5)
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))

print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))

print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
f,ax=plt.subplots(2, 2, figsize=(20,15))

sns.countplot('Embarked', data=df_train, ax=ax[0,0])

ax[0,0].set_title('(1) No. Of Passengers Boarded')

sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax[0,1])

ax[0,1].set_title('(2) Male-Female Split for Embarked')

sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1,0])

ax[1,0].set_title('(3) Embarked vs Survived')

sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1,1])

ax[1,1].set_title('(4) Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
df_train["Cabin"].isnull().sum() / df_train.shape[0]
df_train.head()[["PassengerId", "Cabin"]]
df_train["Age"].isnull().sum()
df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

df_test['Initial']= df_test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

#pandas series 에는 data 를 string 으로 바꿔주는 str method,

#거기에 정규표현식을 적용하게 해주는 extract method가 있습니다.

#이를 사용하여 title 을 쉽게 추출할 수 있습니다. title을 Initial column에 저장하겠습니다
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r') 
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
df_train.isnull().sum()[df_train.isnull().sum() > 0]
df_test.isnull().sum()[df_test.isnull().sum() > 0]
print('Embarked has ', sum(df_train['Embarked'].isnull()), ' Null values')
df_train['Embarked'].fillna('S', inplace=True)
df_train.isnull().sum()[df_train.isnull().sum() > 0]
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
df_train.groupby(['Age_cat'])['PassengerId'].count()
df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_train['Embarked'].isnull().any() , df_train['Embarked'].dtypes
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})

df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
heatmap_data = df_train[['Survived', 'Pclass', 'Sex','Age_cat', 'Age']] 



colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,

           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})



del heatmap_data
#모델의 성능을 높이기 위해 one-hot encoding을 해줄 수 있습니다.

df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')
df_train.head()
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_train.head()
df_train.dtypes
#importing all the required ML packages

from sklearn.ensemble import RandomForestClassifier # 유명한 randomforestclassfier 입니다. 

from sklearn import metrics # 모델의 평가를 위해서 씁니다

from sklearn.model_selection import train_test_split # traning set을 쉽게 나눠주는 함수입니다.
X_train = df_train.drop('Survived', axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values
X_train.shape, X_test.shape
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.2, random_state=2018)
y_tr.shape, y_vld.shape
model = RandomForestClassifier()#생성

model.fit(X_tr, y_tr)           #학습

prediction = model.predict(X_vld)#예측 단 3줄로 가능하다.
print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
from sklearn.linear_model import LogisticRegression

model2 = LogisticRegression()

model2.fit(X_tr, y_tr)

prediction = model2.predict(X_vld)
print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
from pandas import Series



feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')

plt.show()