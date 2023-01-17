# import packages

import pandas as pd
# data read

gen_sub = pd.read_csv("../input/titanic/gender_submission.csv")

gen_sub
train = pd.read_csv("../input/titanic/train.csv")

train
test = pd.read_csv("../input/titanic/test.csv")

test
import sys

# import pandas as pd

import numpy as np

import scipy as sp

import matplotlib

import sklearn



# 버전 체크

print("Python version: {}". format(sys.version))

print("pandas version: {}". format(pd.__version__))

print("NumPy version: {}". format(np.__version__))

print("SciPy version: {}". format(sp.__version__)) 

print("matplotlib version: {}". format(matplotlib.__version__))

print("scikit-learn version: {}". format(sklearn.__version__))



# 보조 라이브러리

import random

import time



import warnings # 오류 무시

warnings.filterwarnings("ignore")





print("-"*25)
#흔히 사용되는 모델 알고리즘

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#흔히 쓰이는 모델 보조 툴들

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



#시각화

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

# from pandas.tools.plotting import scatter_matrix

# 이건 왜 안될까?



#시각화 기본값 설정

#%matplotlib inline = 그림을 주피터 노트북 내부에서 띄울 수 있도록 설정해주는 코드

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

plt.style.use('seaborn')

sns.set(font_scale=2.5)

pylab.rcParams['figure.figsize'] = 12,8
gen_sub
train
test
# train 데이터 null 값 확인



for col in train.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (train[col].isnull().sum() / train[col].shape[0]))

    print(msg)
# test 데이터 null 값 확인



for col in test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (test[col].isnull().sum() / test[col].shape[0]))

    print(msg)
# missingno 라이브러리 호출

import missingno as msno



msno.matrix(df=train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
msno.bar(df=train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
train['Initial']= train.Name.str.extract('([A-Za-z]+)\.')

    

test['Initial']= test.Name.str.extract('([A-Za-z]+)\.')



# Checking the Initials with the Sex

pd.crosstab(train['Initial'], train['Sex']).T.style.background_gradient(cmap='summer_r')
# 위 결과에 근거하여 이름을 수정하자.



train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)



test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
# 각 이니셜 별 평균값 확인

train.groupby('Initial').mean()
train.groupby('Initial')['Survived'].mean().plot.bar()
train.loc[(train.Age.isnull())&(train.Initial=='Mr'),'Age'] = 33

train.loc[(train.Age.isnull())&(train.Initial=='Mrs'),'Age'] = 36

train.loc[(train.Age.isnull())&(train.Initial=='Master'),'Age'] = 5

train.loc[(train.Age.isnull())&(train.Initial=='Miss'),'Age'] = 22

train.loc[(train.Age.isnull())&(train.Initial=='Other'),'Age'] = 46



test.loc[(test.Age.isnull())&(test.Initial=='Mr'),'Age'] = 33

test.loc[(test.Age.isnull())&(test.Initial=='Mrs'),'Age'] = 36

test.loc[(test.Age.isnull())&(test.Initial=='Master'),'Age'] = 5

test.loc[(test.Age.isnull())&(test.Initial=='Miss'),'Age'] = 22

test.loc[(test.Age.isnull())&(test.Initial=='Other'),'Age'] = 46
print('Embarked has ', sum(train['Embarked'].isnull()), ' Null values')
# Embarked에는 2개의 결측값만 있다.

# S에 가장 많은 탑승객이 있으므로 간단하게 S로 채우도록 하자.

train['Embarked'].fillna('S', inplace=True)
test.loc[test.Fare.isnull(), 'Fare'] = test['Fare'].mean()
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1 # 자신을 포함해야하니 1을 더하자.

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1 # 자신을 포함해야하니 1을 더하자.
# 1. loc을 사용(노가다가 많음)



train['Age_cat'] = 0

train.loc[train['Age'] < 10, 'Age_cat'] = 0

train.loc[(10 <= train['Age']) & (train['Age'] < 20), 'Age_cat'] = 1

train.loc[(20 <= train['Age']) & (train['Age'] < 30), 'Age_cat'] = 2

train.loc[(30 <= train['Age']) & (train['Age'] < 40), 'Age_cat'] = 3

train.loc[(40 <= train['Age']) & (train['Age'] < 50), 'Age_cat'] = 4

train.loc[(50 <= train['Age']) & (train['Age'] < 60), 'Age_cat'] = 5

train.loc[(60 <= train['Age']) & (train['Age'] < 70), 'Age_cat'] = 6

train.loc[70 <= train['Age'], 'Age_cat'] = 7



test['Age_cat'] = 0

test.loc[test['Age'] < 10, 'Age_cat'] = 0

test.loc[(10 <= test['Age']) & (test['Age'] < 20), 'Age_cat'] = 1

test.loc[(20 <= test['Age']) & (test['Age'] < 30), 'Age_cat'] = 2

test.loc[(30 <= test['Age']) & (test['Age'] < 40), 'Age_cat'] = 3

test.loc[(40 <= test['Age']) & (test['Age'] < 50), 'Age_cat'] = 4

test.loc[(50 <= test['Age']) & (test['Age'] < 60), 'Age_cat'] = 5

test.loc[(60 <= test['Age']) & (test['Age'] < 70), 'Age_cat'] = 6

test.loc[70 <= test['Age'], 'Age_cat'] = 7
# 2. 간단한 함수로 만들어 apply 메소드 활용

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

    

train['Age_cat_2'] = train['Age'].apply(category_age)
# 두 방법을 비교

print('1번 방법, 2번 방법 둘다 같은 결과를 내면 True -> ', (train['Age_cat'] == train['Age_cat_2']).all())
train.drop(['Age_cat_2'], axis=1, inplace=True)
# Initial 수치화

train['Initial'] = train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

test['Initial'] = test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
# Embarked 수치화 전, 어떤 값이 있는지 확인 먼저 해보기

train['Embarked'].unique()
# Embarked 수치화

train['Embarked'] = train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

test['Embarked'] = test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
# Sex 수치화

train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})

test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})
train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)

test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)
# train 데이터 확인

train.head()
# test 데이터 확인

test.head()
f, ax = plt.subplots(1, 2, figsize=(18, 8))



train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=train, ax=ax[1])

ax[1].set_title('Count plot - Survived')



plt.show()
# Pclass별 승객 수 확인

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
# Pclass별 생존한 승객 수 확인

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
# 위 두 과정을 합친 결과

pd.crosstab(train['Pclass'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
# 각 클래스별 생존 확률 도표화

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
# 위 결과들을 종합하여 보자.



y_position = 1.02

f, ax = plt.subplots(1, 2, figsize=(18, 8))

train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])

ax[0].set_title('Number of Passengers By Pclass', y=y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()
f, ax = plt.subplots(1, 2, figsize=(18, 8))

train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(train['Sex'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass', 'Survived', hue='Sex', data=train, 

               size=6, aspect=1.5)
sns.factorplot(x='Sex', y='Survived', col='Pclass',

               data=train, satureation=.5, size=9, aspect=1)
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(train['Age'].max()))

print('제일 어린 탑승객 : {:.1f} Years'.format(train['Age'].min()))

print('탑승객 평균 나이 : {:.1f} Years'.format(train['Age'].mean()))
# Age에 따른 히스토그램

fig, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.kdeplot(train[train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(train[train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
# Age distribution withing classes

plt.figure(figsize=(8, 6))

train['Age'][train['Pclass'] == 1].plot(kind='kde')

train['Age'][train['Pclass'] == 2].plot(kind='kde')

train['Age'][train['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])
cummulate_survival_ratio = []

for i in range(1, 80):

    cummulate_survival_ratio.append(train[train['Age'] < i]['Survived'].sum() / len(train[train['Age'] < i]['Survived']))

    

plt.figure(figsize=(7, 7))

plt.plot(cummulate_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=train, scale='count', split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=train, scale='count', split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
# Age 분석이 끝났으니 변수는 지워버리자!



train.drop(['Age'], axis=1, inplace=True)

test.drop(['Age'], axis=1, inplace=True)
f, ax = plt.subplots(1, 1, figsize=(7, 7))

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
f,ax=plt.subplots(2, 2, figsize=(20,15))

sns.countplot('Embarked', data=train, ax=ax[0,0])

ax[0,0].set_title('(1) No. Of Passengers Boarded')

sns.countplot('Embarked', hue='Sex', data=train, ax=ax[0,1])

ax[0,1].set_title('(2) Male-Female Split for Embarked')

sns.countplot('Embarked', hue='Survived', data=train, ax=ax[1,0])

ax[1,0].set_title('(3) Embarked vs Survived')

sns.countplot('Embarked', hue='Pclass', data=train, ax=ax[1,1])

ax[1,1].set_title('(4) Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
print("Maximum size of Family: ", train['FamilySize'].max())

print("Minimum size of Family: ", train['FamilySize'].min())
f,ax=plt.subplots(1, 3, figsize=(40,10))

sns.countplot('FamilySize', data=train, ax=ax[0])

ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)



sns.countplot('FamilySize', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)



train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(train['Fare'], color='b', label='Skewness : {:.2f}'.format(train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
train['Fare'] = train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

test['Fare'] = test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(train['Fare'], color='b', label='Skewness : {:.2f}'.format(train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
heatmap_data = train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']] 



colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,

           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})



del heatmap_data
# 모든 분석이 끝났으니 원-핫 인코딩을 진행하자!



# Initial One-hot encoding

train = pd.get_dummies(train, columns=['Initial'], prefix='Initial')

test = pd.get_dummies(test, columns=['Initial'], prefix='Initial')



# Embarked One-hot encoding

train = pd.get_dummies(train, columns=['Embarked'], prefix='Embarked')

test = pd.get_dummies(test, columns=['Embarked'], prefix='Embarked')

# Sex는 이미 0과 1로만 이루어져있으니 원핫인코딩 제외
# 최종 train 데이터 확인

train
# 최종 test 데이터 확인

test
X_train = train.drop('Survived', axis=1).values

target_label = train['Survived'].values

X_test = test.values
X_tr, X_vld, y_tr, y_vld = model_selection.train_test_split(X_train, target_label, test_size=0.3, random_state=2018)
model = ensemble.RandomForestClassifier()

model.fit(X_tr, y_tr)

prediction = model.predict(X_vld)
print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
from pandas import Series



feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=test.columns)



plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')

plt.show()
gen_sub
prediction = model.predict(X_test)

gen_sub['Survived'] = prediction

gen_sub.to_csv('./my_first_submission.csv', index=False)