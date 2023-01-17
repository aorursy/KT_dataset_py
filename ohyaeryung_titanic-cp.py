import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn



plt.style.use('seaborn')

sns.set(font_scale=2.5) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib의 기본 scheme말고 seaborn scheme을 세팅하고, 일일이 graph의 font size를 지정할 필요 없이 seaborn의 font_scale을 사용하면 편합니다.

import missingno as msno # 데이터 셋에 채워지지 않는 널 데이터를 쉽게 보여줄 수 있는 라이브러리



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline 

# notebook을 실행한 브라우저에서 바로 그림을 볼 수 있게 해주는 것
os.listdir("../input") # 하위 디렉토리
train_data = pd.read_csv('../input/train.csv')

train_data.head()
test_data = pd.read_csv('../input/test.csv')

test_data.head()
train_data.columns
train_data.isnull().sum()
for col in train_data.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (train_data[col].isnull().sum() / train_data[col].shape[0]))

    print(msg)
for col in test_data.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (test_data[col].isnull().sum() / test_data[col].shape[0]))

    print(msg)
f, ax = plt.subplots(1, 2, figsize=(18, 8)) # pyplot에 subplot을 그려줌, 1행 2열



train_data['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

# value_counts() 개수를 반환해줌, explode는 원을 짼다, autopct는 %의미, ax[0]는 첫번째 그림

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('') # y축 레이블을 없애겠다는 의미

sns.countplot('Survived', data=train_data, ax=ax[1])

ax[1].set_title('Count plot - Survived')



plt.show()
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()

# count는 각각의 객체가 샘플이 몇 명이 있느냐
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()

# sum은 숫자 자체(데이터)를 더한 것 
pd.crosstab(train_data['Pclass'], train_data['Survived'], margins=True).style.background_gradient(cmap='summer_r')

# margine은 all생성, gradient는 기울어지는 형태에 따라 background 색을 바꿔주겠다, colormaps=cmap 간격에 따라 색깔을 줘서 보기 편하게 만듬
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()

# as index는 True하면 Pclass를 인덱스로 두느냐(false로 적으면 plot선이 두개가 그려져서 안됨,, ascending은 내림차순으로 바꿔줌
y_position = 1.02

f, ax = plt.subplots(1, 2, figsize=(18, 8))

train_data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])

ax[0].set_title('Number of Passengers By Pclass', y=y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass', hue='Survived', data=train_data, ax=ax[1])

# hue로 색깔 구분해서 나타냄(파,주)

ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()
f, ax = plt.subplots(1, 2, figsize=(18, 8))

train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=train_data, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# mean함수는 평균, sort_values를 호출하면 해당 변수에 대해 정렬
pd.crosstab(train_data['Sex'], train_data['Survived'], margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass', 'Survived', hue='Sex', data=train_data,size=6, aspect=1.5)
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(train_data['Age'].max()))

print('제일 어린 탑승객 : {:.1f} Years'.format(train_data['Age'].min()))

print('탑승객 평균 나이 : {:.1f} Years'.format(train_data['Age'].mean()))
#sort the ages into logical categories

train_data['Age'] = train_data['Age'].fillna(-0.5)

test_data['Age'] = test_data['Age'].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train_data["AgeGroup"] = pd.cut(train_data['Age'], bins, labels = labels)

test_data["AgeGroup"] = pd.cut(test_data['Age'], bins, labels = labels)



#draw a bar plot of Age vs. survival

plt.rcParams["figure.figsize"] = (20, 8)

sns.barplot(x="AgeGroup", y="Survived", data=train_data)

plt.show()
# Age distribution within classes

plt.figure(figsize=(8, 6))

train_data['Age'][train_data['Pclass'] == 1].plot(kind='kde')

train_data['Age'][train_data['Pclass'] == 2].plot(kind='kde')

train_data['Age'][train_data['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])
f, ax = plt.subplots(1, 1, figsize=(7, 7))

train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
f,ax=plt.subplots(2, 2, figsize=(20,15)) # 2 by 2 그림

sns.countplot('Embarked', data=train_data, ax=ax[0,0]) # 2차원 [0,0]

ax[0,0].set_title('(1) No. Of Passengers Boarded')

sns.countplot('Embarked', hue='Sex', data=train_data, ax=ax[0,1])

ax[0,1].set_title('(2) Male-Female Split for Embarked')

sns.countplot('Embarked', hue='Survived', data=train_data, ax=ax[1,0])

ax[1,0].set_title('(3) Embarked vs Survived')

sns.countplot('Embarked', hue='Pclass', data=train_data, ax=ax[1,1])

ax[1,1].set_title('(4) Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2, hspace=0.5) # 각 plot간의 간격들(좌우간격, 상하간격을 맞춰줌)

plt.show()
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
print("Maximum size of Family: ", train_data['FamilySize'].max())

print("Minimum size of Family: ", train_data['FamilySize'].min())
f,ax=plt.subplots(1, 3, figsize=(40,10))

sns.countplot('FamilySize', data=train_data, ax=ax[0])

ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)



sns.countplot('FamilySize', hue='Survived', data=train_data, ax=ax[1])

ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)



train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(train_data['Fare'], color='b', label='Skewness : {:.2f}'.format(train_data['Fare'].skew()),ax=ax)

# displot은 series에 바르면? 시리즈2속 그림을 그려주는 plot, label=Skewness는 왜도(평균을 중심으로 한쪽으로 치우져져 있는 경향) 

g = g.legend(loc='best') # g는 matplot라이브러리 객체
test_data.loc[test_data.Fare.isnull(), 'Fare'] = test_data['Fare'].mean() # testset 에 있는 nan value 를 평균값으로 치환합니다.



train_data['Fare'] = train_data['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

test_data['Fare'] = test_data['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(train_data['Fare'], color='b', label='Skewness : {:.2f}'.format(train_data['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
train_data.head()
def get_one_hot(array):

    return np.array((array['Pclass'] == 1, array['Pclass'] == 2,

                    array['Pclass'] == 3, array['Sex'] == 'male',

                    array['Sex'] == 'female', array['SibSp'],

                    array['Parch'], array['Fare'],

                    array['Embarked'] == 'C', array['Embarked'] == 'Q',

                    array['Embarked'] == 'S')).swapaxes(0, 1).astype('float32')
x_train = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

x_train.head()
x_train = get_one_hot(x_train)



x_train[:10]
y_train = np.array(train_data['Survived'])



y_train[:10]
x_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]



x_test.head()
x_test = get_one_hot(x_test)



x_test[:10]
train_data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis=1, inplace=True)

test_data.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis=1, inplace=True)
train_data.head()
test_data.head()
from mlxtend.classifier import StackingCVClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp = imp.fit(x_train)

x_train_imp = imp.transform(x_train)
clf1 = RandomForestClassifier()

clf2 = GradientBoostingClassifier()

lr = LogisticRegression()



sclf = StackingCVClassifier(classifiers=[clf1, clf2], meta_classifier=lr)
param_test = {'randomforestclassifier__n_estimators': [10, 120],

              'randomforestclassifier__max_depth': [2, 15],

              'gradientboostingclassifier__n_estimators': [10, 120],

              'gradientboostingclassifier__max_depth': [2, 15],

              'gradientboostingclassifier__learning_rate' : [0.01, 0.1],

              'meta_classifier__C': [0.1, 10.0]}
sclf.fit(x_train_imp, y_train)
x = ['Pclass:1', 'Pclass:2', 'Pclass:3', 'male', 'female', 'SibSp', 'Parch', 'Fare', 'Embarked:C', 'Embarked:Q', 'Embarked:S']



plt.figure(figsize=(38, 10))

plt.title('Importance for the classification')

plt.bar(x, sclf.clfs_[0].feature_importances_)

plt.show()
X_test_imp = imp.transform(x_test)
submission = pd.read_csv('../input/sample_submission.csv')
# data = np.array([np.array(test_data['PassengerId']), sclf.predict(x_test_imp)]).swapaxes(0, 1)



submission = pd.DataFrame(data, columns=['PassengerId', 'Survived'])

submission.set_index('PassengerId', inplace=True)



submission.head()
prediction = sclf.predict(x_test)

submission['Survived'] = prediction

prediction
submission.to_csv('./submission.csv',index= False)