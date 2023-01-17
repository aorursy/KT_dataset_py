# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')

sns.set(font_scale=2.5)

import missingno as msno



# ignore warning

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
## Data Load - Pandas 사용 및 shape, head 및 columns methods
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.describe()
# train, test 데이터 shape

df_train.shape, df_test.shape
'''

column(feature)는 총 12개로 이루어져 있으며

학습에 사용해야 할 feature는 11개이다.

예측해야 할 feature는 survived이다.

'''



columns = df_train.columns

columns
df_train.head()
df_test.head()
df_train.dtypes # dtype들을 보여준다
df_train.describe()
df_test.describe()
df_train.isnull().sum()
df_train.shape[0]
df_train.isnull().sum() / df_train.shape[0]
# test set도 확인해보자

df_test.isnull().sum() / df_test.shape[0]
df_train['Survived'].value_counts()
'''

target label의 분포가 제법 균일(balanced)하다.

불균일한 경우, 예를 들어 100중 1이 99, 0이 1개인 경우에는 

만약 모델이 모든 것을 1이라고 해도 정확도는 99%가 된다.

0을 찾는 문제라면 이 모델은 원하는 결과를 줄 수 없게 된다



지금 문제에서는 문제되지 않기에 계속 진행한다

'''
# Pclass 그룹 별 데이터 카운트

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()

# Pclassdhk Survived 데이터를 가져온 뒤 Pclass로 groupby를 한다

# class의 그룹마다 숫자를 카운팅 하기 위해 count()함수를 사용한다
# Pclass 그룹 별 생존자 수 합

df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).sum()
# 위와 같은 작업을 crosstab로 편하게 할 수도 있다

pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True)
# mean은 생존률을 구하게 할 수 있습니다.

df_train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=True).mean()
# 이를 시각화 해보겠다

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().plot.bar()
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train[['Sex','Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('sex: Survived vs Dead')

plt.show()
# sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train,

#                size=6, aspect=1.5)



sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, 

               size=6, aspect=1.5)
# print('Oldest : {:.1f} Years'.format(df_train['Age'].max()))

# print('Youngest : {:.1f} Years'.format(df_train['Age'].min()))

# print('Average : {:.1f} Years'.format(df_train['Age'].mean()))





print('Oldest : {:.1f} Years'.format(df_train['Age'].max()))

print('Youngest : {:.1f} Years'.format(df_train['Age'].min()))

print('Average : {:.1f} Years'.format(df_train['Age'].mean()))
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
# Age distribution withing classes

plt.figure(figsize=(8, 6))

df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])
df_train['Embarked'].unique()

f, ax = plt.subplots(1, 1, figsize=(7, 7))

df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
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

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
print("Maximum size of Family: ", df_train['FamilySize'].max())

print("Minimum size of Family: ", df_train['FamilySize'].min())

f,ax=plt.subplots(1, 3, figsize=(40,10))

sns.countplot('FamilySize', data=df_train, ax=ax[0])

ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)



sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
df_test.isnull().sum() / df_test.shape[0]

#df_test.isnull().sum() / len(df_test)
# 특이하기도 train set 말고 test set에 Fare 피쳐에 널 값이 하나 존재하는 것을 확인할 수 있었습니다.

# 그래서 평균 값으로 해당 널값을 넣어줍니다.

df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean() # testset 에 있는 nan value 를 평균값으로 치환합니다.
df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
df_train["Cabin"].isnull().sum() / df_train.shape[0]
df_train.head()[["PassengerId", "Cabin"]]

df_train['Ticket'].value_counts()

df_train["Age"].isnull().sum()

df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

df_test['Initial']= df_test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex
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
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})

df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat', 'Age']] 



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
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_train.head()

df_train.dtypes

df_test.head()

df_test.dtypes

#import all the required ML packages



from sklearn.linear_model import LogisticRegression

from sklearn import metrics # 모델 평가에 사용된다

from sklearn.model_selection import train_test_split # training set을 쉽게 나눠주는함수이다
X_train = df_train.drop('Survived', axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values
X_train.shape, X_test.shape
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.2, random_state=2018)

y_tr.shape, y_vld.shape
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier(random_state=201) # clf(classifier)를 DecisionTree로 설정해준다

clf.fit(X_tr, y_tr)

clf_prediction = clf.predict(X_vld)



from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
print("Accuracy is ", accuracy_score(clf_prediction, y_vld))
print(classification_report(clf_prediction, y_vld))
confusion_matrix(clf_prediction, y_vld)

from sklearn.model_selection import GridSearchCV



parameters = {'min_samples_split': range(5,300,20),'max_depth': range(1,20,2)}

clf_tree=DecisionTreeClassifier()

clf=GridSearchCV(clf_tree,parameters)

clf
clf.fit(X_tr,y_tr)

clf_prediction = clf.predict(X_vld)
print("Accuracy is ", accuracy_score(clf_prediction, y_vld))

print(classification_report(clf_prediction, y_vld))
confusion_matrix(clf_prediction, y_vld)
