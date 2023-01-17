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



plt.style.use('bmh') # matplotlib style gallery 

sns.set(font_scale=2.5)



import missingno as msno



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline 

#새로운 윈도우 창으로 보여지지 않도록 해줌.
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train.head()
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
df_train.head()

# feature에 대한 특징 파악
df_train.describe()

# numerical한 변수들의 통계적 기본 특징
df_test.describe()

# 테스트 데이터에 대한 설명 

# 이 데이터프레임은 Survived를 예측해야 되기 때문에 하나의 feature가 적게 보인다.
df_train.columns

# df_train에 있는 컬럼들을 모두 나타낸다.

# 이것을 for문을 적용
for col in df_train.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)

# null 데이터가 얼마나 있는지

# 셀에서 F 키를 누른 상태로 df_train과 df_train를 바꿔줄 수 있다.
for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))

    print(msg)
msno.matrix(df=df_train.iloc[:, :], figsize=(8,8),color=(0.5,0.8,0.1))

# color는 RGB 숫자를 의미한다.

# 그래프상 빈칸은 null값을 의미한다.
msno.bar(df=df_train.iloc[:, :], figsize=(8,8),color=(0.5,0.8,0.1))

# 위와 같이 null data를 파악하는 것인데, 시각화 막대그래프 형태
f, ax = plt.subplots(1,2,figsize=(18,8))



df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', ax=ax[0], shadow=True)

# ax는 0 죽은사람과 1 산사람을 나타낸다.

ax[0].set_title('Pie plot - Survived') # 제목 달아주기

ax[0].set_ylabel('') # y축의 라벨 - 파이그래프 왼쪽에 나타난다.

sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived')



plt.show()

# 생존과 죽음을 비율로 나타낸 파이그래프

# 카운트한 막대그래프 두가지로 나타내보기
df_train.shape

# 인덱스 제외하고 총 11개의 feature가 있다.

# 어떤 feature이 중요할까?
df_train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=True).count()
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).sum()
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')

# margins는 'All'값을 의미한다.

# cmap은 검색에 'color example code' 검색
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar()

# 생존율을 나타내는 것. 0은 죽음, 1은 생존이라면 이것들의 (0 * x + 1 * y) / N를 나타냄
y_position = 1.02

f, ax = plt.subplots(1,2,figsize=(18,8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number of Passengers By Pclass', y=y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass',hue='Survived',data=df_train,ax=ax[1])

ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0]) 

# as_index는 컬럼을 'Sex'로 만드는 것 ??

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)
# 아래 두가지 중 위의 것이 참고용

# 아래 중 윗쪽은 어느 클래스든 여성이 생존율이 높다. 특히 Pclass의 기울기가 더 크므로.

# 아래 중 아래쪽은 클래스별의 차이를 보여주기에 용이한듯 

sns.factorplot(x='Sex', y='Survived', col='Pclass',

              data=df_train, satureation=.5,

               size=9, aspect=1

              )
sns.factorplot(x='Pclass', y='Survived', col='Sex',

              data=df_train, satureation=.5,

               size=9, aspect=1

              )
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))

print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))

print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))
# 히스토그램에 대한 설명

df_train[df_train['Survived'] == 1]['Age'].hist()
# 생존, 사망별 Age의 분포(Kde plot)

fig, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

# boolean을 indexing하는 방법을 잘 숙지해야 한다.

# df_train['Survived'] == 1 이라는 것 중 True인 row만 반환해서 'Age'컬럼을 뽑아내는 것

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
# Age distribution withing classes

plt.figure(figsize=(8, 6)) # figsize는 도화지를 의미한다.

df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

# kind를 hist로 하면 그래프가 겹쳐서 안보인다.



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])
# 설명엔 나타나있지 않은 것

# 위에선 생존에 관한 kde를 볼 수 없으므로, boolean을 이용해서 생존자별 나이대의 분포를 알아본다.

# Class별로 생존자 나이대의 분포를 살펴보자.

fig, ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[(df_train['Survived'] == 0) & (df_train['Pclass'] == 1 )]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived'] == 1) & (df_train['Pclass'] == 1 )]['Age'], ax=ax)

plt.legend(['Survived == 0', 'Survived == 1'])

plt.title('1st Class')

plt.show()
fig, ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[(df_train['Survived'] == 0) & (df_train['Pclass'] == 2 )]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived'] == 1) & (df_train['Pclass'] == 2 )]['Age'], ax=ax)

plt.legend(['Survived == 0', 'Survived == 1'])

plt.title('2st Class')

plt.show()
fig, ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(df_train[(df_train['Survived'] == 0) & (df_train['Pclass'] == 3 )]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived'] == 1) & (df_train['Pclass'] == 3 )]['Age'], ax=ax)

plt.legend(['Survived == 0', 'Survived == 1'])

plt.title('3st Class')

plt.show()
cummulate_survival_ratio = []

for i in range(1, 80):

    cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))

# 전체 인원 수 중에서(len) survived한 사람들을 (1) sum한 것들의 비율을 반환

    

plt.figure(figsize=(7, 7))

plt.plot(cummulate_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=df_train, scale='count', split=True,ax=ax[0])

# scale을 count로 하면, 0과 1에서 상대적으로 더 많은 곳의 넓이가 커 보인다.(분포 + 실제 비율)

# scale을 area로 하면, 균등한 넓이로 비교하게 해준다.(동일한 조건에서 비율만 보기 위함)

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=df_train, scale='count', split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
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

# pandas 시리즈끼리는 연산이 가능하다
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
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean() # testset 에 있는 nan value 를 평균값으로 치환합니다.

# 평균대치법

df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

# 훈련용 데이터와 학습용 데이터의 변경된 값을 다시 배정
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
df_train['Ticket'].value_counts()
df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')

df_test['Initial'] = df_test.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')

# 추출한 Initial과 Sex간의 count 확인
# 남자 성, 여자 성을 위의 카운트를 참고하여 적용시키고 replace 메소드를 이용하여 치환



df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)



df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
df_train.groupby('Initial').mean()
# Miss와 Mrs가 생존률이 높음을 보여주는 그래프

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
print('Embarked has', sum(df_train['Embarked'].isnull()), 'Null values')
df_train['Embarked'].fillna('S', inplace=True)

# fillna는 NA값을 채우는 메소드이다.
# 첫번째 방법인 loc를 사용해보자.

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
# 두번째 방법인 apply를 사용해보자. 훨씬 간단하다.

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

# 앞의 방법과 동일한 결과를 냈는지 비교하기 위해서 다른 컬럼을 생성
# all 메소드를 통해서 두 Series가 동일한 Boolean을 갖는지 비교해준다. 단 하나라도 동일하지 않으면 False를 반영한다.

print('1번 방법, 2번 방법 둘다 같은 결과를 내면 True 줘야함 -> ', (df_train['Age_cat'] == df_train['Age_cat_2']).all())
# 동일한 값을 갖는것을 확인 한 후, 분석을 위해 Age와 Age_cat_2를 제거한다.

df_train.drop(['Age', 'Age_cat_2'], axis=1, inplace=True)

df_test.drop(['Age'], axis=1, inplace=True)
# Initial 변수 변경

df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
# Embarked 변수 변경 전 데이터 확인

df_train['Embarked'].unique() # 혹은 .value_counts()
# 데이터가 세가지인것을 확인 한 후 변경

df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
# 변경이 올바르게 되었는지 확인

# isnull() 메소드는 하나라도 null이 있으면 True를 반영한다.

# any() 메소드는 하나라도 True가 있으면 True를 반영한다.

# 즉, 하나라도 null이 있으면 True를 반영하기 때문에  null이 없다면 False를 반영해야 정상이다.

df_train['Embarked'].isnull().any()
# Sex도 이산형으로 변경 

df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})

df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
# heatmap plot으로 상관관계를 시각화



heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']]



colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={'size':16})



del heatmap_data
df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')
df_train.head() # 올바르게 되었는지 확인
# Embarked에도 적용

df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
df_train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis=1, inplace=True)
df_train.head()
df_test.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics # 모델 평가

from sklearn.model_selection import train_test_split # training set을 나눠주는 함수 
X_train = df_train.drop('Survived', axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)
model = RandomForestClassifier()

model.fit(X_tr, y_tr)

prediction = model.predict(X_vld)
print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100*metrics.accuracy_score(prediction, y_vld)))
from pandas import Series



feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8,8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')

plt.show()
submission = pd.read_csv("../input/titanic/gender_submission.csv")
submission.head()
prediction = model.predict(X_test)

submission['Survived'] = prediction
submission.to_csv("./my_first_submission.csv", index=False)