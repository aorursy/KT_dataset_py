!pwd
!cat /proc/cpuinfo

# 가상컴퓨터의 정보를 나타냄.
!nvidia-smi
# colab - google drive mount

from google.colab import drive

drive.mount('gdrive')

!pwd

# /content에서 작업하면 작업내용 다 삭제 된다. 
!ls -al 

%cd "/content/gdrive/My Drive/4th"

!ls -al
!ls -al ./dataset
# kaggle 설치

#!pip install kaggle
# upload kaggle.json

#from google.colab import files

#files.upload()
#!mkdir -p ~/.kaggle

#!cp kaggle.json ~/.kaggle

#!kaggle competitions list
# 다운로드 전에, 해당 competition에 join 해야 함

#!kaggle competitions download -c titanic
#!mv train.csv test.csv gender_submission.csv "/content/drive/My Drive/GoogleColab/kaggle_titanic/dataset/"
#!ls -al 

#!mkdir ./datasets

#!mv *.csv ./datasets/
#!head -5 ./datasets/gender_submission.csv

#!tail -f ./datasets/gender_submission.csv

#!wc -l ./datasets/gender_submission.csv

#!head -5 ./datasets/train.csv

#!tail -f ./datasets/train.csv

#!wc -l ./datasets/train.csv

#!head -5 ./datasets/test.csv

#!tail -f ./datasets/test.csv

#!wc -l ./datasets/test.csv
!pip install missingno
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# seaborn scheme 설정

plt.style.use('seaborn')

# 그래프의 폰트 설정

sns.set(font_scale=2.5) 

# 데이터셋의 missing data 쉽게 보여주기

import missingno as msno



#ignore warnings

#import warnings

#warnings.filterwarnings('ignore')



%matplotlib inline
WORK_DIR = './'

df_train = pd.read_csv(WORK_DIR + '/datasets/train.csv')

df_test = pd.read_csv(WORK_DIR + '/datasets/test.csv')
# 데이터 셋 살펴보기

df_train.head()
df_test.head()
# 통계적 수치 보기

df_train.describe()
df_test.describe()
# 학습 데이터 체크

for col in df_train.columns:

    print('column: {:>10}\t Percent of NULL value: {:.2f}%'.format(col, 

          100 * (df_train[col].isnull().sum() / df_train[col].shape[0])))

df_train.info()
# 테스트 데이터 체크

for col in df_test.columns:

    print('column: {:>10}\t Percent of NULL value: {:.2f}%'.format(col, 

          100 * (df_test[col].isnull().sum() / df_test[col].shape[0])))
df_test.info()
# null data 분포 확인

msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
# null data 수로 확인

msno.bar(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
df_train['Survived'].value_counts()
# 1행 2열 팔레트, 크기(세로:18, 가로:8)

f, ax = plt.subplots(1, 2, figsize=(18, 8))



# 파이 차트로 그리기

# value_counts() 의 data type은 series이며,

# series 타입은 plot을 가짐

# plt.plot(df_train['Survived'].value_counts()) 은 df_train[..]...plot()과 동일

df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], 

                           autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')



sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived')



plt.show()
df_train['Survived'].value_counts().plot()
plt.plot(df_train['Survived'].value_counts())
# 11개의 feature, 1개의 target label 

df_train.shape
df_train.shape[0]
# Pclass 별 항목 갯수

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
# Pclass별 생존자 수

# P1(136/216), P2(87/184), P3(119/491)

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
# crosstab 으로 확인

pd.crosstab(df_train['Pclass'], df_train['Survived'], 

            margins=True).style.background_gradient(cmap='summer_r')
# 클래스별 생존률

# P1 : (136 / (80+136)) => 63%

df_train[['Pclass', 'Survived']].groupby(['Pclass'], 

             as_index=True).mean().sort_values(by='Survived', 

                                   ascending=False).plot.bar()
# label에 따른 갯수 확인

y_position = 1.02

f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Pclass'].value_counts().plot.bar(

    color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])

ax[0].set_title('Number of Passengers By Pclass', y=y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train[['Sex', 'Survived']].groupby(['Sex'], 

                          as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
df_train[['Sex', 'Survived']].groupby(['Sex'], 

            as_index=False).mean().sort_values(by='Survived', ascending=False)
# crosstab 으로 확인

pd.crosstab(df_train['Sex'], df_train['Survived'], 

            margins=True).style.background_gradient(cmap='summer_r')
# 3개의 차원 데이터로 이루어진 그래프 그리기

sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, 

               size=6, aspect=1.5)
# cloumn 대신 hue 사용

sns.factorplot(x='Sex', y='Survived', col='Pclass',

              data=df_train, satureation=.5,

               size=9, aspect=1)

sns.factorplot(x='Sex', y='Survived', hue='Pclass',

              data=df_train, satureation=.5,

               size=9, aspect=1)
# 간단한 통계 보기

print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))

print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))

print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))
# 생존에 따른 Age의 히스토그램

# kdeplot()

fig, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
# 히스토그램 vs. kdeplot()

# kdeplot()이 부드럽게 그림

# (참고) 커널밀도추정 https://blog.naver.com/loiu870422/220660847923

df_train[df_train['Survived']==1]['Age'].hist()
# pandas indexing

df_train.iloc[0,:]
for row in df_train.iterrows():

  break

row
df_train['Survived'] == 1
df_train[df_train['Survived']==1]
# figsize

# 아래 세 예제는 동일

#f = plt.figure(figsize=(10,10))

#f, ax = plt.subplots(1,1,figsize=(10,10))

#plt.figure(figsize=(10,10))

f, ax = plt.subplots(1,1,figsize=(5,5))

a = np.arange(100)

b = np.sin(a)

ax.plot(b)



plt.figure(figsize=(5,5))

plt.plot(b)
# Pclass와 Age 로 확인

plt.figure(figsize=(8, 6))

df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])
fig, ax = plt.subplots(1,3,figsize=(20,5))

sns.kdeplot(df_train[(df_train['Survived']==0) & (df_train['Pclass']==1)]['Age'], ax=ax[0])

sns.kdeplot(df_train[(df_train['Survived']==1) & (df_train['Pclass']==1)]['Age'], ax=ax[0])

ax[0].set_title('1st class')

ax[0].legend(['Survived==0', 'Survived==1'])   

sns.kdeplot(df_train[(df_train['Survived']==0) & (df_train['Pclass']==2)]['Age'], ax=ax[1])

sns.kdeplot(df_train[(df_train['Survived']==1) & (df_train['Pclass']==2)]['Age'], ax=ax[1])

ax[1].set_title('2nc class')

ax[1].legend(['Survived==0', 'Survived==1'])   

sns.kdeplot(df_train[(df_train['Survived']==0) & (df_train['Pclass']==3)]['Age'], ax=ax[2])

sns.kdeplot(df_train[(df_train['Survived']==1) & (df_train['Pclass']==3)]['Age'], ax=ax[2])

ax[2].set_title('3rd class')

ax[2].legend(['Survived==0', 'Survived==1'])                               

plt.show()

# 나이 범위에 따른 생존률

cummulate_survival_ratio = []

for i in range(1, 80):

    cummulate_survival_ratio.append(

        df_train[df_train['Age'] < i]['Survived'].sum() / 

        len(df_train[df_train['Age'] < i]['Survived']))

    

plt.figure(figsize=(7, 7))

plt.plot(cummulate_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
# scale='count', scale='area'

f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=df_train, scale='count', 

               split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=df_train, scale='count', 

               split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
f, ax = plt.subplots(1, 1, figsize=(7, 7))

df_train[['Embarked', 'Survived']].groupby(['Embarked'], 

              as_index=True).mean().sort_values(by='Survived', 

                                      ascending=False).plot.bar(ax=ax)
# 다른 feature로 split하여 확인

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
# 새로운 컬럼(Family) 추가

# series 타입은 서로 더할 수 있음

# 자신을 포함하기 위해 1을 더함

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 



print("Maximum size of Family: ", df_train['FamilySize'].max())

print("Minimum size of Family: ", df_train['FamilySize'].min())
# Family 크기와 생존 관계

f,ax=plt.subplots(1, 3, figsize=(40,10))

sns.countplot('FamilySize', data=df_train, ax=ax[0])

ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)



sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], 

                    as_index=True).mean().sort_values(by='Survived', 

                                         ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
# histogram

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(df_train['Fare'], color='b', 

                 label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
# NULL값 치환

df_train.loc[df_train.Fare.isnull(), 'Fare'] = df_train['Fare'].mean()

df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()



df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i>0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i>0 else 0)



fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(df_train['Fare'], color='b', 

            label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
df_train.head()
df_train['Ticket'].value_counts()

df_train['Age'].isnull().sum()
df_train['Name']
df_train['Name'].str.extract('([A-Za-z]+)\.')
# initial 항목으로 추출

df_train['Initial']=0

for i in df_train:

    df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.') 

    

df_test['Initial']=0

for i in df_test:

    df_test['Initial']= df_test.Name.str.extract('([A-Za-z]+)\.')
# Sex와 Initial에 대한 crosstab 확인

pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')
# 위 테이블을 참고하여,

# initial 치환

df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',

                          'Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other',

                       'Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)



df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',

                          'Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other',

                         'Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')
df_train.groupby('Initial').mean()
# 생존률 확인

df_train.groupby('Initial')['Survived'].mean().plot.bar()
df_all = pd.concat([df_train, df_test])

df_all
df_all.reset_index(drop=True)
df_all.groupby('Initial').mean()
df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mr'),'Age']=33

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mrs'),'Age']=37

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Master'),'Age']=5

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Miss'),'Age']=22

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Other'),'Age']=45



df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mr'),'Age']=33

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mrs'),'Age']=37

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Master'),'Age']=5

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Miss'),'Age']=22

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Other'),'Age']=45
df_train['Age'].isnull().sum()
df_test['Age'].isnull().sum()
df_train['Embarked'].isnull().sum()
df_train.shape
df_train['Embarked'].fillna('S', inplace=True)
df_train['Age_cat'] = 0



df_train.head()
# loc 이용

# 10살 간격으로 나누기

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
df_train.head()
# apply() 함수 사용한 방법

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

df_train.head()
# 두 가지 방법의 비교

# all() : 모두 True 일 때, True

# any() : 하나라도 True이면 True

(df_train['Age_cat'] == df_train['Age_cat_2']).all()
# Age 컬럼 삭제

# axis=1

df_train.drop(['Age', 'Age_cat_2'], axis=1, inplace=True)

df_test.drop(['Age'], axis=1, inplace=True)
df_train.Initial.unique()
df_train['Initial'] = df_train['Initial'].map(

    {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_test['Initial'] = df_test['Initial'].map(

    {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_train.Initial.unique() 
df_train.Initial.unique()
df_train['Embarked'].unique()
df_train['Embarked'].value_counts()
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_train.head()
# null 확인

df_train['Embarked'].isnull().any()
df_train['Sex'].unique()
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})

df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
df_train['Sex'].unique()
heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 

                         'Embarked', 'FamilySize', 'Initial', 'Age_cat']] 



colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,

           square=True, cmap=colormap, linecolor='white', annot=True, 

            annot_kws={"size": 16})



del heatmap_data
df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')



df_train.head()
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
df_train.head()
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_train.head()
df_test.head()
from sklearn.ensemble import RandomForestClassifier  

from sklearn import metrics 

from sklearn.model_selection import train_test_split
# 학습에 쓰일 데이터와 target label 분리

X_train = df_train.drop('Survived', axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)
# 학습

model = RandomForestClassifier()

model.fit(X_tr, y_tr)

# 예측

prediction = model.predict(X_vld)
prediction
# 정확도

print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
print(X_tr.shape)

print(X_vld.shape)
(prediction == y_vld).sum()/prediction.shape[0]
model.feature_importances_
df_train.head()

from pandas import Series

feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')

plt.show()
WORK_DIR = './'

submission = pd.read_csv(WORK_DIR + '/datasets/sample_submission.csv')
submission.head()
prediction = model.predict(X_test)

submission['Survived'] = prediction



submission.to_csv('./titanic_submission.csv', index=False)
!head -20 ./titanic_submission.csv
from google.colab import files

files.download("./titanic_submission.csv")

#!kaggle competitions submit -c titanic -f titanic_submission.csv -m t20191210_01