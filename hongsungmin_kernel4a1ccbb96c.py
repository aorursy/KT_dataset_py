import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



plt.style.use('seaborn')    # plt seaborn 스타일

sns.set(font_scale=2.5)      # size 2.5로 통일



import missingno as msno    # data에 채워지지 않은 것을 채우기 위해 import



import warnings

warnings.filterwarnings('ignore')





%matplotlib inline

# 비어 있는 데이터에 대해서 처리를 먼저 해주어야 한다.



df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
df_train.head(5)
df_train.describe() # 각 feature에 대해 count, mean, std, min ... max 를 알려준다.

                    # count의 경우 숫자가 그냥 갯수를 세는데 숫자가 적은 경우 NULL 값이 있다는 것을 말한다.
df_test.describe()
for col in df_train.columns:

    msg = 'column: {:>10}\t Percent of NaNValue: {:.2f}%'.format(col, 100*(df_train[col].isnull().sum()/df_train[col].shape[0])) # {:>10} 오른쪽 정렬 의미

                                # df_train[col]은 pandas의 시리즈 type .isnull null 이면

                                # isnull = True or False 인데 sum으로 True를 1로 다더함

    print(msg)
for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaNValue: {:.2f}%'.format(col, 100*(df_test[col].isnull().sum()/df_test[col].shape[0])) # {:>10} 오른쪽 정렬 의미

                                # df_train[col]은 pandas의 시리즈 type .isnull null 이면

                                # isnull = True or False 인데 sum으로 True를 1로 다더함

    print(msg)
msno.matrix(df=df_train.iloc[:, :], figsize=(8,8), color=(0.8, 0.5, 0.2))

# iloc은 panda의 문법으로 col, row 범위를 설정해 잘라주는 것, figsize 크기, color RGB색깔
f, ax = plt.subplots(1, 2, figsize=(18, 8))

#

df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

# explode 원판 간격 째는거, autopct %형식을 말해주는 것, ax = ax[i], shadow

# plt.plot(df_train['Survived'].value_counts())

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=df_train, ax=ax[1])

# Column, 어떤 데이터 셋, 도화지의 두번째

ax[1].set_title('Count plot - Survived')

plt.show()
print(df_train['Survived'].value_counts()) # series 가 어떻게 구성되어있는지 feature와 해당 갯수, 데이터 Type 구해줌
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).count()

# as_index = False -> 기존에 Pclass 자체가 index로 사용이 되는데 index를 따로 만들어서 써라!!

# Pclass로 Survived를 묶는다, Count : Pclass가 1인 Survivied의 갯수
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).sum()

# df_train['Survived'].unique() # array([0, 1])

# Pclass로 Survivied를 묶는다, Sum : Pclass는 0 또는 1 이다 이것의 합을 의미 -> 실제 생존자수

# .mean() -> sum()에 대한 평균을 의미 -> 각 Pclass별 생존율을 알 수 있음
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).count()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False).plot()

# 끝에 .plot()을 붙이면 그래프 형식 안붙이면 표 형식

# .sort_values(by = '칼럼', ascending = 오름차순 True, 내림차순 False)
y_position = 1.02

f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax[0])

# 단순히 Pclass 갯수를 보여주는 bar 그래프가 된다.

ax[0].set_title('Number of passengers by Pcalss', y=y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])

# Pcalss 별로 분류하지만 hue = 'Survived'로 Survived에 대해서 각 분류결과를 더 자세히 보여줌

ax[1].set_title('Pclass : Survived vs Dead', y=y_position)

plt.show()
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived Sex')

sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

# sns.countplot 1.첫번째 기준 2.두번째 기준 3.data= 4.ax=

ax[1].set_title('Sex : Survived vs Dead')

plt.show()
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()

# 평균을 구하는 순간 성별의 생존확률이된다. 왜일까 
pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')

# margins 은 합친 결과를 보여주냐 마느냐 Column All이 추가된다!
# Both Sex and Pclass
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)

# factor plot   X축        Y축        hue 색깔      data=        size=    aspect=
sns.factorplot(x='Sex', y='Survived', col='Pclass', data=df_train, saturation=.5, size=9, aspect=1)

# col을쓰냐 hue를 쓰냐에 따라 한그림에서 그래프를 볼수도있고 col 처럼 다르게 볼수도 있음!!
# Age
print('나이 max : {:.1f} years'.format(df_train['Age'].max()))

print('나이 min : {:.1f} years'.format(df_train['Age'].min()))

print('나이 평균 : {:.1f} years'.format(df_train['Age'].mean()))

fig, ax = plt.subplots(1, 1, figsize=(9,5))

sns.kdeplot(df_train[df_train['Survived']==1]['Age'], ax=ax)

# df_train[df_train['Survived']==1] 이렇게 bool 값을 넣게 되면 True에 해당하는 값만 시리즈에서 가져오게 된다.

# 이때 ['Age']를 붙이지 않으면 다가져오는거고 ['Age']를 붙이면 Age column에 해당하는 값을 가져오는거!!

sns.kdeplot(df_train[df_train['Survived']==0]['Age'], ax=ax)

plt.legend(['Survived==1', 'Survived==0'])

plt.show()
plt.figure(figsize=(8, 6))

df_train['Age'][df_train['Pclass']==1].plot(kind='kde')

df_train['Age'][df_train['Pclass']==2].plot(kind='kde')

df_train['Age'][df_train['Pclass']==3].plot(kind='kde')

# [X축][Y축]

plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3th Class'])

# plt는 xlabel, ax는 set_xlabel 이다!
flg, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.kdeplot(df_train[(df_train['Survived']==1)&(df_train['Pclass']==1)]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived']==0)&(df_train['Pclass']==1)]['Age'], ax=ax)

plt.legend(['Survived 1', 'Survived 0'])

plt.show()
change_age_range_survive_ratio = []

for i in range(0, 80):

    change_age_range_survive_ratio.append(df_train[df_train['Age']<i]['Survived'].sum() / len(df_train[df_train['Age']<i]['Survived']))

plt.figure(figsize=(7, 7))

plt.plot(change_age_range_survive_ratio)

# plot은 어떤 리스트(배열)을 도화지에 놓는 기능이라고 보면 되겠다!

plt.title('Survived rate change depend on range of Age', y=1.02)

plt.ylabel('Survived rate')

plt.xlabel('Range of Age 0~X')

plt.show()

# Pclass, Age, Survived
# seaborn의 바이올린plot?

f, ax = plt.subplots(1, 2, figsize=(18, 8))

sns.violinplot('Pclass', 'Age', hue='Survived', data=df_train, scale='count', split=True, ax=ax[0])

# scale='area'의 경우 영역의 넓이를 동일학 출력한다

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0, 110, 10))

# y좌표 Age값 0~110, 10단위로 끊어서 출력



sns.violinplot('Sex', 'Age', hue='Survived', data=df_train, scale='count', split=True, ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0, 110, 10))

plt.show()
# Embarked
f, ax = plt.subplots(1, 1, figsize=(7, 7))

df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived').plot.bar(ax=ax)

#sort_index() 도 가능하다 Embarked : C S Q 순으로 정렬됨
f, ax = plt.subplots(2, 2, figsize=(20, 15))

sns.countplot('Embarked', data=df_train, ax=ax[0, 0])

ax[0, 0].set_title('(1) No. of Passengers Boared')



sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax[0, 1])

ax[0, 1].set_title('(2) No. Male Female split for Embarked')



sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1, 0])

ax[1, 0].set_title('(3) No. Survived vs Embarked')



sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1, 1])

ax[1, 1].set_title('(4) No. Pclass vs Embarked')



plt.subplots_adjust(wspace=0.2, hspace=0.5)

# subplots_adjust wspace, hspace subplot사이간 w, h

plt.show()
# SibSp : 형제,자매,배우자 Parch : 자식들
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1

#위와 같이 Pandas series는 연산이 가능하다 (+, -, *, / ...)
f, ax = plt.subplots(1, 3, figsize=(40,10))

sns.countplot('FamilySize', data=df_train, ax=ax[0])

ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)

sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('(2) No. Survived countplot depending of FamilySize', y=1.02)

df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) No. Survived rate depending on FamilySize', y=1.02)

#sort_values는 ( by='', ascending=T or F) 가 들어간다!

plt.subplots_adjust(hspace=0.2, wspace=0.5)

plt.show()
# DEA : Fare 탑승요금 continous, Cabin, Ticket
flt, ax = plt.subplots(1, 1, figsize=(8, 7))

g = sns.distplot(df_train['Fare'],color='b', label='SKewness:{:.2f}'.format(df_train['Fare'].skew()), ax=ax)

#distplot : 시리즈를 넣으면 히스토그램 그려주는 함수, skew() : 히스토그램이 얼마나 쏠렸냐, 비대칭이냐!

g = g.legend(loc='best')

#SKewness : 왜도 -> 양수면 좌측으로 쏠린거고, 음수면 우측

#뾰족한정도 : 첨도 -> 양수면 뾰족, 음수면 평평, 0은 가우시안?
#df_train[Fare]에 값이 0보다 크면 log를 취한것을 maping한다

df_train['Fare'] = df_train['Fare'].map(lambda i : np.log(i) if i > 0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i : np.log(i) if i > 0 else 0)

#Fare값 자체에 log를 씌워줌으서 Skewness를 줄여준다.
flt, ax = plt.subplots(1, 1, figsize=(8, 7))

g = sns.distplot(df_train['Fare'],color='b', label='SKewness:{:.2f}'.format(df_train['Fare'].skew()), ax=ax)

#distplot : 시리즈를 넣으면 히스토그램 그려주는 함수, skew() : 히스토그램이 얼마나 쏠렸냐, 비대칭이냐!

g = g.legend(loc='best')
df_train['Ticket'].value_counts() #value_counts 예시 항상 볼 것
#Feature engineering Fill Null in Age
df_train['Age'].isnull().sum() # 177개의 Null

df_train['Name'].str # String으로 형변환됨

df_train['Name'].str.extract('([A-Za-z]*)\.') # character한개이상 뒤에 . 이 붙은것 Mr, Mrs, Miss ...

df_train['Initial'] = df_train['Name'].str.extract('([A-Za-z]*)\.')

df_test['Initial'] = df_test['Name'].str.extract('([A-Za-z]*)\.')

pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

#replace로 Initial 통일, inplace=True는 바로 replace 해주겠다

df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_train.groupby(['Initial']).mean()
df_train.groupby(['Initial'])['Survived'].mean().plot.bar()
df_all = pd.concat([df_train, df_test]) # pd.concat 트레인 셋과 테스트 셋을 합침

df_all.reset_index() # 기존index로 그대로 합쳐지는데 합친뒤 index를 새로 매김
df_all.groupby(['Initial']).mean()
# df_train.loc[1, :]  # 두번째 row의 전체 column을 가져온다.

# df_train.loc[df_train['Survived']==1] # Survived가 1인 것을 모두 들고온다.

df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial']=='Mr'), 'Age']=33

df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial']=='Mrs'), 'Age']=37

df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial']=='Miss'), 'Age']=21

df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial']=='Master'), 'Age']=5

df_train.loc[(df_train['Age'].isnull()) & (df_train['Initial']=='Other'), 'Age']=45

# loc 활용하는거 보기!

df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial']=='Mr'), 'Age']=33

df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial']=='Mrs'), 'Age']=37

df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial']=='Miss'), 'Age']=21

df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial']=='Master'), 'Age']=5

df_test.loc[(df_test['Age'].isnull()) & (df_test['Initial']=='Other'), 'Age']=45

# df_train['Age'].isnull().sum()

df_test['Age'].isnull().sum()
# Embarked

df_train['Embarked'].isnull().sum()
df_train['Embarked'].fillna('S', inplace=True)

df_test['Fare'].fillna(0, inplace=True)
df_train['Age_cat'] = 0

df_train.head()
df_train.loc[df_train['Age']<10, 'Age_cat'] = 0

df_train.loc[(df_train['Age']>=10)&(df_train['Age']<20), 'Age_cat'] = 1

df_train.loc[(df_train['Age']>=20)&(df_train['Age']<30), 'Age_cat'] = 2

df_train.loc[(df_train['Age']>=30)&(df_train['Age']<40), 'Age_cat'] = 3

df_train.loc[(df_train['Age']>=40)&(df_train['Age']<50), 'Age_cat'] = 4

df_train.loc[(df_train['Age']>=50)&(df_train['Age']<60), 'Age_cat'] = 5

df_train.loc[(df_train['Age']>=60)&(df_train['Age']<70), 'Age_cat'] = 6

df_train.loc[df_train['Age']>=70, 'Age_cat'] = 7

df_train.head()
#함수로 하는 방식은 .apply(함수) 로 적용!

def category_Age(x):

    if x<10:

        return 0

    if x<20:

        return 1

    if x<30:

        return 2

    if x<40:

        return 3

    if x<50:

        return 4

    if x<60:

        return 5

    if x<70:

        return 6

    else:

        return 7
df_test.loc[df_test['Age']<10, 'Age_cat'] = 0

df_test.loc[(df_test['Age']>=10)&(df_test['Age']<20), 'Age_cat'] = 1

df_test.loc[(df_test['Age']>=20)&(df_test['Age']<30), 'Age_cat'] = 2

df_test.loc[(df_test['Age']>=30)&(df_test['Age']<40), 'Age_cat'] = 3

df_test.loc[(df_test['Age']>=40)&(df_test['Age']<50), 'Age_cat'] = 4

df_test.loc[(df_test['Age']>=50)&(df_test['Age']<60), 'Age_cat'] = 5

df_test.loc[(df_test['Age']>=60)&(df_test['Age']<70), 'Age_cat'] = 6

df_test.loc[df_test['Age']>=70, 'Age_cat'] = 7
#Age_cat Column을 만들었으므로 Age Column은 날린다

df_train.drop(['Age'], axis=1, inplace=True)

df_test.drop(['Age'], axis=1, inplace=True)

df_train.head()
#String Feature Change!! int로 바꾸겠지~
df_train['Initial'].unique()

df_train.Embarked.value_counts()
df_train['Initial'] = df_train['Initial'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4})

df_test['Initial'] = df_test['Initial'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4})

#map을 이용해
df_train['Embarked'] = df_train['Embarked'].map({'C':0, 'Q':1, 'S':2})

df_test['Embarked'] = df_test['Embarked'].map({'C':0, 'Q':1, 'S':2})

df_train.Embarked.isnull().sum()
df_train.Sex.unique()
df_train.Sex = df_train.Sex.map({'male':1, 'female':0})

df_test.Sex = df_test.Sex.map({'male':1, 'female':0})
df_train.head()
## heatmap

heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']]

heatmap_data.corr()
colormap = plt.cm.RdBu

# matplotlib cm

plt.figure(figsize=(10, 10))

plt.title('Pearson Correalation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, 

            square=True, cmap=colormap, linecolor='White', annot=True)

# astype 형변환, corr은 변수간 상관관계 다 구해주는 것! -> 0 선형관계없음, 양수 비례, 음수 반비례

# lineweidths 선굵기, linecolor 선색, annot 네모에 숫자표기

# 프로젝트에서 중점적으로 다뤄야하는 것이 될수도!

#one hot encoding
df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')

df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

#prefix를 쓰면 Initial Column이 날라간다.

df_test.head()
df_train.head()
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_train.head()

df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.head()
from sklearn.ensemble import RandomForestClassifier #랜덤포레스트

from sklearn import metrics                         #

from sklearn.model_selection import train_test_split#Train set, Valid set, Test set -> Train, Valid 나눠주는 기능

df_train.shape
X_train = df_train.drop('Survived', axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values

X_train.shape
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)

#test_size = Valid로 30%를 주고 70%f를 Train으로 준다.

X_tr.shape
model = RandomForestClassifier()

model.fit(X_tr, y_tr) # X train과 y train으로 학습을 시킨다 랜덤 포레스트
prediction = model.predict(X_vld)

X_vld.shape
prediction
print('총 {}명 중 {:.2f}% 정확도로 생존 맞춤'.format(y_vld.shape[0], 100*metrics.accuracy_score(prediction, y_vld)))

# y_vld.shape[0] = 268명, metrics.acc.. (prediction, y_vld) 비교
(prediction == y_vld).sum()/y_vld.shape[0]

# Ture, False 배열 쭉 나열됨 0.83582
from pandas import Series

model.feature_importances_

# sklearn은 학습 시킨 model에 대하여 feature_importances_ 를 가지고 있다.
#학습시킨 모델에서 결과와 가장 관계가있는 feature

feature_importances = model.feature_importances_

plt.figure(figsize=(8,8))

Series_feat_imp = Series(feature_importances, index = df_test.columns)

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')

plt.show()
#이제 진짜 test 셋을 이용해보자

submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission.head()
prediction = model.predict(X_test)
submission['Survived'] = prediction
submission.to_csv('./my_first_submission2.csv', index=False)