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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split

from pandas import Series

# 데이터셋의 missing data 쉽게 보여주기

import missingno as msno



# seaborn scheme 설정

plt.style.use('seaborn')

# 그래프 폰트 설정

sns.set(font_scale=1.5)



# ignore warnings

warnings.filterwarnings('ignore')



# %matplotlib inline



'''Dataset 확인'''

df_train = pd.read_csv('train.csv')

df_test = pd.read_csv('test.csv')



# 데이터 셋 살펴보기

print(df_train.head())

print(df_test.head())



# 통계적 수치 보기

print(df_train.describe())

print(df_test.describe())



'''NULL 데이터 체크'''

print('\n1\n')

# 학습 데이터 체크

for col in df_train.columns:

    print('column: {:>10}\t Percent of NULL value: {:.2f}%'.format(col, 100 * (

            df_train[col].isnull().sum() / df_train[col].shape[0])))



print('\n2\n')

print(df_train.info())

print('\n3\n')

# 테스트 데이터 체크

for col in df_test.columns:

    print('column: {:>10}\t Percent of NULL value: {:.2f}%'.format(col, 100 * (

            df_test[col].isnull().sum() / df_test[col].shape[0])))



print('\n4\n')

print(df_test.info())



'''MSNO 라이브러리를 사용하여 null data 확인'''

print('\n5\n')

print(df_train.shape)



# null data 분포 확인

msno.matrix(df=df_train.iloc[:, :], color=(1.0, 0.75, 0.8))

# plt.show()



# null data 수로 확인

msno.bar(df_train.iloc[:, :], color=(0.25, 0.24, 0.27))

# plt.show()



'''Target Lable 확인'''

print('\n6\n')

print(df_train['Survived'].value_counts())



# 1행 2열 팔레트 , 크기 (세로:18,가로:8)

f, ax = plt.subplots(1, 2, figsize=(16, 8))



# 파이 차트로 그리기

# value_counts()의 data type은 series이며,

# series 타입은 plot을 가짐

# plt.plot(df_train['Survived'].value_counts()) 은 df_train[..]...plot()과 동일

df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)



ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')



sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived')



# plt.show()





df_train['Survived'].value_counts().plot()

plt.plot(df_train['Survived'].value_counts())



'''Exploratory Data Analysis(EDA, 탐색적 데이터 분석)'''



# 11개의 feature,1개의 target label

print('\n7\n')

print(df_train.shape)

## Pclass(클래스)

# Pclass 별 항목 갯수

print('\n8\n')

print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count())



# Pclass 별 생존자 수

# P1(136/216), P2(87/184), P3(119/491)

print('\n9\n')

print(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum())  # mean



# crosstab 으로 확인

print('\n9.1\n')

print(pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True))



# 클래스별 생존률

# P1 : (136/( 80+136))=> 63%

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived',

                                                                                       ascending=False).plot.bar()

# label에 따른 갯수 확인

y_position = 1.02

f_1, ax_1 = plt.subplots(1, 2)

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#D3D3D3'], ax=ax_1[0])



ax_1[0].set_title('Number of Passengers By Pclass', y=y_position)

ax_1[0].set_ylabel('Count')

sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax_1[1])

ax_1[1].set_title('Pclass: Survived vs Dead', y=y_position)



## 성별(Sex)

f_2, ax_2 = plt.subplots(1, 2)

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax_2[0])

ax_2[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=df_train, ax=ax_2[1])

ax_2[1].set_title('Sex: Survived vs Dead')



print('\n10\n')

print(df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))



# crosstab 으로 확인

print('\n10.1\n')

print(pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True))



## Pclass 와 Sex

# 3개의 차원 데이터로 이루어진 그래프 그리기

sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train)



# cloumn 대신 hue 사용

sns.factorplot(x='Sex', y='Survived', col='Pclass', data=df_train, satureation=.5, size=9, aspect=1)

sns.factorplot(x='Sex', y='Survived', hue='Pclass', data=df_train, satureation=.5, size=9, aspect=1)



## Age

# 간단한 통계 보기

print('\n11\n')

print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))

print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))

print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))



print('\n12\n')

print(df_train.info())



print('\n13\n')

print(df_train[df_train['Survived'] == 1]['Age'].isnull().sum())



# 생존에 따른 Age의 히스토그램

# kdeplot()

f_3, ax_3 = plt.subplots(1, 1)

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'].dropna(), ax=ax_3)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'].dropna(), ax=ax_3)

plt.legend(['Survived == 1', 'Survived == 0'])



# 히스토그램 vs kdeplot()

# kdeplot()이 부드럽게 그림

# (참고) 커널밀도추정 https://blog.naver.com/loiu870422/220660847923

f_4, ax_4 = plt.subplots(1, 1)

df_train[df_train['Survived'] == 1]['Age'].hist()



# pandas indexing

print('\n14\n')

print(df_train.iloc[0, :])

# 위아래 똑같은 결과임

print('\n15\n')

for row in df_train.iterrows():

    print(row)

    break



print('\n16\n')

print(df_train['Survived'] == 1)



print('\n17\n')

print(df_train[df_train['Survived'] == 1])



# Pclass와 Age로 확인

plt.figure()

df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])



f_5, ax_5 = plt.subplots(1, 3)

sns.kdeplot(df_train[(df_train['Survived'] == 0) & (df_train['Pclass'] == 1)]['Age'].dropna(), ax=ax_5[0])

sns.kdeplot(df_train[(df_train['Survived'] == 1) & (df_train['Pclass'] == 1)]['Age'].dropna(), ax=ax_5[0])

ax_5[0].set_title('1st class')

ax_5[0].legend(['Survived==0', 'Survived==1'])

sns.kdeplot(df_train[(df_train['Survived'] == 0) & (df_train['Pclass'] == 2)]['Age'].dropna(), ax=ax_5[1])

sns.kdeplot(df_train[(df_train['Survived'] == 1) & (df_train['Pclass'] == 2)]['Age'].dropna(), ax=ax_5[1])

ax_5[1].set_title('2nd class')

ax_5[1].legend(['Survived==0', 'Survived==1'])

sns.kdeplot(df_train[(df_train['Survived'] == 0) & (df_train['Pclass'] == 3)]['Age'].dropna(), ax=ax_5[2])

sns.kdeplot(df_train[(df_train['Survived'] == 1) & (df_train['Pclass'] == 3)]['Age'].dropna(), ax=ax_5[2])

ax_5[2].set_title('3rd class')

ax_5[2].legend(['Survived==0', 'Survived==1'])



# 나이 범위에 따른 생존률

cummulate_survival_ratio = []

for i in range(0, 81):

    cummulate_survival_ratio.append(

        df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))



plt.figure(figsize=(7, 7))

plt.plot(cummulate_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')



## Pclass, Sex, Age

# scale='count' , scale='area'

f_6, ax_6 = plt.subplots(1, 2)

sns.violinplot('Pclass', 'Age', hue='Survived', data=df_train, scale='count', split=True, ax=ax_6[0])

ax_6[0].set_title('Pclass and Age vs Survived')

ax_6[0].set_yticks(range(0, 110, 10))

sns.violinplot("Sex", "Age", hue="Survived", data=df_train, scale='count', split=True, ax=ax_6[1])

ax_6[1].set_title('Sex and Age vs Survived')

ax_6[1].set_yticks(range(0, 110, 10))



##Embarked

f_7, ax_7 = plt.subplots(1, 1)

df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived',

                                                                                           ascending=False).plot.bar(

    ax=ax_7)



# 다른 feature로 split하여 확인

f_8, ax_8 = plt.subplots(2, 2, figsize=(20, 15))



sns.countplot('Embarked', data=df_train, ax=ax_8[0, 0])

ax_8[0, 0].set_title('(1) No. Of Passengers Boarded')



sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax_8[0, 1])

ax_8[0, 1].set_title('(2) Male-Female Split for Embarked')



sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax_8[1, 0])

ax_8[1, 0].set_title('(3) Embarked vs Survived')



sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax_8[1, 1])

ax_8[1, 1].set_title('(4) Embarked vs Pclass')



plt.subplots_adjust(wspace=0.2, hspace=0.5)



##Family

# 새로운 컬럼(Family) 추가

# series 타입은 서로 더할 수 있음

# 자신을 포함하기 위해 1을 더함

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1



print('\n18\n')

print("Maximum size of Family: ", df_train['FamilySize'].max())

print("Minimum size of Family: ", df_train['FamilySize'].min())



# Family 크기와 생존 관계

f_9, ax_9 = plt.subplots(1, 3, figsize=(40, 10))



sns.countplot('FamilySize', data=df_train, ax=ax_9[0])

ax_9[0].set_title('(1) No. Of Passengers Boarded', y=1.02)



sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax_9[1])

ax_9[1].set_title('(2) Survived countplot depending on FamilySize', y=1.02)



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'],

                                             as_index=True).mean().sort_values(by='Survived',

                                                                               ascending=False).plot.bar(ax=ax_9[2])

ax_9[2].set_title('(3) Survived rate depending on FamilySize', y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)



##Fare(탑승요금)

# histogram

fig_10, ax_10 = plt.subplots(1, 1)

g = sns.distplot(df_train['Fare'], color='b',

                 label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax_10)

g = g.legend(loc='best')



print('\n19\n')

print(df_train.info())



# NULL값 치환

df_train.loc[df_train.Fare.isnull(), 'Fare'] = df_train['Fare'].mean()  #################치환하는거 다른방식은?

df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()



# log 적용 (편향된 데이터 보정)

df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)



f_11, ax_11 = plt.subplots(1, 1)

g_1 = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax_11)

g_1 = g_1.legend(loc='best')



##Cabin

print('\n20\n')

print(df_train.head())



##Ticket

print('\n21\n')

print(df_train['Ticket'].value_counts())



'''Feature Engineering'''  # # # # # # # # # # # # # # # # # # # null 값 처리하는 부분이니까 일단 여기서 바꿀 수 있는게 있을 듯

##Age의 NULL처리

print('\n22\n')

print(df_train['Age'].isnull().sum())



print('\n23\n')

print(df_train['Name'])



print('\n24\n')

print(df_train['Name'].str.extract('([A-Za-z]+)\.'))



# initial 항목으로 추출

df_train['Initial'] = 0

df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')



df_test['Initial'] = 0

df_test['Initial'] = df_test.Name.str.extract('([A-Za-z]+)\.')



# Sex와 Initial에 대한 crosstab 확인

print('\n25\n')

print(pd.crosstab(df_train['Initial'], df_train['Sex']).T)



# 위 테이블을 참고하여,

# initial 치환

df_train['Initial'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess',

                             'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don', 'Dona'],

                            ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other',

                             'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mr'], inplace=True)



df_test['Initial'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess',

                            'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don', 'Dona'],

                           ['Miss', 'Miss', 'Miss', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other',

                            'Other', 'Other', 'Mr', 'Mr', 'Mr', 'Mr'], inplace=True)



print('\n26\n')

print(pd.crosstab(df_train['Initial'], df_train['Sex']).T)



print('\n27\n')

print(df_train.groupby('Initial').mean())



# 생존률 확인

f_12, ax_12 = plt.subplots(1, 1)

df_train.groupby('Initial')['Survived'].mean().plot.bar()



# train,test 전체 셋을 사용하여 Age의 null값 처리

print('\n27\n')

print(df_train.head(5))



print('\n28\n')

df_all = pd.concat([df_train, df_test])

print(df_all.head(10))

print('\n29\n')

print(df_all.tail(5))



df_all.reset_index(drop=True)

print('\n30\n')

print(df_all.head())

print('\n31\n')

print(df_all.groupby('Initial').mean())



df_list = [df_train, df_test]

for df in df_list:

    df.loc[(df.Age.isnull()) & (df.Initial == 'Mr'), 'Age'] = 33

    df.loc[(df.Age.isnull()) & (df.Initial == 'Mrs'), 'Age'] = 37

    df.loc[(df.Age.isnull()) & (df.Initial == 'Master'), 'Age'] = 5

    df.loc[(df.Age.isnull()) & (df.Initial == 'Miss'), 'Age'] = 22

    df.loc[(df.Age.isnull()) & (df.Initial == 'Other'), 'Age'] = 45



print('\n32\n')

print(df_train['Age'].isnull().sum())

print(df_test['Age'].isnull().sum())



# Embarked의 Null값 처리

print('\n33\n')

print(df_train['Embarked'].isnull().sum())

print('\n34\n')

print(df_train.shape)

df_train['Embarked'].fillna('S', inplace=True) # 날린줄 알았더니 왜... 살아있지...

# df_train['Embarked'].dropna()



# Age 변환

# df_train['Age_cat'] = 0

print('\n35\n')

print(df_train.head())





# apply() 함수 사용한 방법

def category_age(x):

    if 0 <= x < 5:

        return 0

    elif 5 <= x < 9:

        return 1

    elif 9 <= x < 16:

        return 2

    elif 16 <= x < 27:

        return 3

    elif 27 <= x < 40:

        return 4

    elif 40 <= x < 48:

        return 5

    elif 48 <= x < 58:

        return 6

    elif 58 <= x < 75:

        return 7

    else:

        return 8



    # if 0 <= x < 10:

    #     return 0

    # elif 10 <= x < 20:

    #     return 1

    # elif 20 <= x < 30:

    #     return 2

    # elif 30 <= x < 40:

    #     return 3

    # elif 40 <= x < 50:

    #     return 4

    # elif 50 <= x < 60:

    #     return 5

    # elif 60 <= x < 70:

    #     return 6

    # else:

    #     return 7



    # if 0 <= x < 10:

    #     return 0

    # elif 10 <= x < 16:

    #     return 1

    # elif 16 <= x < 26:

    #     return 2

    # elif 26 <= x < 33:

    #     return 3

    # elif 33 <= x < 43:

    #     return 4

    # elif 43 <= x < 50:

    #     return 5

    # elif 50 <= x < 55:

    #     return 6

    # elif 55 <= x < 76:

    #     return 7

    # else:

    #     return 8





df_train['Age_cat_2'] = df_train['Age'].apply(category_age)

df_test['Age_cat_2'] = df_test['Age'].apply(category_age)



print('\n36\n')

print(df_train.head())



# Age 컬럼 삭제

# axis=1

df_train.drop(['Age'], axis=1, inplace=True)

df_test.drop(['Age'], axis=1, inplace=True)



# Initial 변경

print('\n37\n')

print(df_train.Initial.unique())



df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

print('\n38\n')

print(df_train.Initial.unique())



# Embarked변경

print('\n39\n')

print(df_train['Embarked'].unique())

print('\n40\n')

print(df_train['Embarked'].value_counts())



df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})



print('\n41\n')

print(df_train.head())



# null 확인

print(df_train['Embarked'].isnull().any())



# sex변경

print('\n42\n')

print(df_train['Sex'].unique())

df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})

df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})



print('\n43\n')

print(df_train['Sex'].unique())



##Person Correlation

heatmap_data = df_train[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Fare', 'SibSp', 'Parch',

                         'Embarked', 'FamilySize', 'Initial', 'Age_cat_2']]



colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,

            square=True, cmap=colormap, linecolor='white', annot=True,

            annot_kws={"size": 16})



del heatmap_data



'''One-Hot-Encoding'''

# Initial , Embarked 을 one-hot-encoding으로 만들기

df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')

# df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

# df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')



'''Drop columns'''  # ## # # # # # # # # # # # # #  이거도 어떻게 수정해 볼 만 한듯

df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'FamilySize'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'FamilySize'], axis=1, inplace=True)



print('\n44\n')

print(df_train.head(), "\n", df_test.head())

print('\n44.1\n')

# 테스트 데이터 체크

for col in df_test.columns:

    print('column: {:>10}\t Percent of NULL value: {:.2f}%'.format(col, 100 * (

            df_test[col].isnull().sum() / df_test[col].shape[0])))



'''모델 만들기'''

## 준비 - 데이터셋을 train,valid,test set 으로 나누기



# 학습의 쓰일 데이터와 target label 분리



X_train = df_train.drop('Survived', axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values



X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.25, random_state=2019)



## 모델 생성 및 예측

# 학습

model = RandomForestClassifier()

model.fit(X_tr, y_tr)



# 예측

prediction_RF = model.predict(X_vld)

print('\n45\n')

print(prediction_RF)



# 정확도

print('\n46\n')

print('@@@@@@@@@@@@@@@@@@@@ 총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0],

                                                                100 * metrics.accuracy_score(prediction_RF, y_vld)))



print('\n47\n')

print(X_tr.shape)

print(X_vld.shape)



print('\n48\n')

print((prediction_RF == y_vld).sum() / prediction_RF.shape[0])



## Feature Importance

print('\n49\n')

print(model.feature_importances_)

print('\n50\n')

print(df_train.head())



feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)



plt.figure()

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')



##Test Set을 사용하여 Prediction

submission = pd.read_csv('sample_submission.csv')

print('\n50\n')

print(submission.head())



# plt.show()

prediction = model.predict(X_test)





##test set에 대하여 예측하고, 결과를 csv에 저장

def predict_titanic(txt, model):

    prediction = model.predict(X_test).astype('uint8')

    submission['Survived'] = prediction

    t = './' + txt + '.csv'

    submission.to_csv(t, index=False)





predict_titanic("titanic_randomforest", model)



# %%time



import lightgbm as lgbm



model_lgbm = lgbm.LGBMClassifier(max_depth=10, lambda_l1=0.1, lambda_l2=0.01, learning_rate=0.15,

                                 n_estimators=500, reg_alpha=1.1, colsample_bytree=0.9, subsample=0.9, n_jobs=12)

model_lgbm.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=False, early_stopping_rounds=50,

               eval_metric="accuracy")

pred_lgbm = model_lgbm.predict(X_vld)

score_lgbm = metrics.accuracy_score(pred_lgbm, y_vld)



print('\n50\n')

print('RandomForest Test Score: ', metrics.accuracy_score(prediction_RF, y_vld))

print("LightGBM Test Score: ", score_lgbm)



predict_titanic("titanic_lightgbm", model_lgbm)



# from pandas import Series

feature_importance = model_lgbm.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)

plt.figure()

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.title("LightGBM")

plt.xlabel('Feature importance')

plt.ylabel('Feature')

# plt.show()





# %%time



import xgboost as xgb



model_xgb = xgb.XGBClassifier(max_depth=7, learning_rate=0.01, n_estimators=500, reg_alpah=1.1,

                              colsample_bytree=0.9, subsample=0.9, n_jobs=12)

model_xgb.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=False, early_stopping_rounds=50)

pred_xgb = model_xgb.predict(X_vld)

score_xgb = metrics.accuracy_score(pred_xgb, y_vld)

print("XGBoost Test score: ", score_xgb)



predict_titanic("titanic_xgboost", model_xgb)



feature_importance = model_xgb.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)

plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.title("XGBoost")

plt.xlabel('Feature importance')

plt.ylabel('Feature')



# %%time



import catboost as cboost



model_cboost = cboost.CatBoostClassifier(depth=10, reg_lambda=0.1, learning_rate=0.01, iterations=500)

model_cboost.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=False, early_stopping_rounds=50)

pred_cboost = model_cboost.predict(X_vld)

score_cboost = metrics.accuracy_score(pred_cboost, y_vld)

print("CatBoost Test Score: ", score_cboost)



predict_titanic("titanic_catboost", model_cboost)



feature_importance = model_cboost.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)

plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.title("CatBoost")

plt.xlabel('Feature importance')

plt.ylabel('Feature')



# %%time



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Activation

from keras.optimizers import Adam

from keras import backend as K



model_mlp = Sequential()

model_mlp.add(Dense(45, activation='relu', input_dim=12))

model_mlp.add(BatchNormalization())



model_mlp.add(Dense(9, activation='relu'))

model_mlp.add(BatchNormalization())

model_mlp.add(Dropout(0.4))



model_mlp.add(Dense(5, activation='relu'))

model_mlp.add(BatchNormalization())

model_mlp.add(Dropout(0.2))



model_mlp.add(Dense(1, activation='sigmoid'))



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)

model_mlp.compile(optimizer=optimizer,

                  loss='binary_crossentropy',

                  metrics=['accuracy'])



hist = model_mlp.fit(X_tr, y_tr, epochs=500, batch_size=90, validation_data=(X_vld, y_vld), verbose=False)



pred_mlp = model_mlp.predict_classes(X_vld)[:, 0]

score_mlp = metrics.accuracy_score(pred_mlp, y_vld)

print("MLP Test Score: ", score_mlp)





##test set에 대하여 예측하고, 결과를 csv에 저장

def predict_titanicneual(txt, model):

    prediction = model.predict_classes(X_test)[:, 0]

    submission['Survived'] = prediction

    t = './' + txt + '.csv'

    submission.to_csv(t, index=False)





predict_titanicneual('titanic_neuralnetwork', model_mlp)



fig, loss_ax = plt.subplots(figsize=(10, 10))



acc_ax = loss_ax.twinx()



loss_ax.plot(hist.history['loss'], 'y', label='train loss')

loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')



acc_ax.plot(hist.history['acc'], 'b', label='train acc')

acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')



loss_ax.set_xlabel('epoch')

loss_ax.set_ylabel('loss')

acc_ax.set_ylabel('accuray')



loss_ax.legend(loc='upper left')

acc_ax.legend(loc='lower left')



plt.show()



# csv파일 5개를 다 합쳐서 다수결로 결과를 다시 만든 다음에 결과를 제출하자



df_RF = pd.read_csv('titanic_randomforest.csv')

df_CB = pd.read_csv('titanic_catboost.csv')

df_LGBM = pd.read_csv('titanic_lightgbm.csv')

df_XGB = pd.read_csv('titanic_xgboost.csv')

df_NN = pd.read_csv('titanic_neuralnetwork.csv')

df_total = []

sum_csv = pd.concat([df_RF, df_CB['Survived'], df_LGBM['Survived'], df_XGB['Survived'], df_NN['Survived']],

                    axis=1)



sum_csv['total'] = df_RF['Survived'] + df_CB['Survived'] + df_LGBM['Survived'] + df_XGB['Survived'] + df_NN[

    'Survived'] > 2

# print(sum_csv.tail(100))



submission['Survived'] = sum_csv['total'].map({False: 0, True: 1})

submission.to_csv('./titanic_total.csv', index=False)



# print('\n51\n')

# print(submission.tail(100))
