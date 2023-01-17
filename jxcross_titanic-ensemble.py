!pip install ipython-autotime
%load_ext autotime
from google.colab import drive
drive.mount("gdrive")

!pwd
!ls -al
%cd "/content/gdrive/My Drive/4th_2020/kisti_kaggle"
!ls -al

import os
os.listdir("./datasets")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn scheme 설정
plt.style.use('seaborn')
# 그래프의 폰트 설정
sns.set(font_scale=1.5) 
# 데이터셋의 missing data 쉽게 보여주기
import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
WORK_DIR = './datasets/'
df_train = pd.read_csv(WORK_DIR + 'train.csv')
df_test = pd.read_csv(WORK_DIR + 'test.csv')
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
df_train.shape
# null data 분포 확인
msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
# null data 수로 확인
msno.bar(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))

df_train['Survived'].value_counts()
# 1행 2열 팔레트, 크기(세로:18, 가로:8)
f, ax = plt.subplots(1, 2, figsize=(16,8))

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
# Pclass 별 항목 갯수
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
# Pclass별 생존자 수
# P1(136/216), P2(87/184), P3(119/491)
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()  # mean()
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
f, ax = plt.subplots(1, 2, figsize=(16, 8))
df_train['Pclass'].value_counts().plot.bar(
    color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
plt.show()
f, ax = plt.subplots(1, 2, figsize=(16, 8))
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
df_train.info()
df_train[df_train['Survived'] == 1]['Age'].isnull().sum()
# 생존에 따른 Age의 히스토그램
# kdeplot()
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'].dropna(), ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'].dropna(), ax=ax)
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
df_train.info()
# NULL값 치환
df_train.loc[df_train.Fare.isnull(), 'Fare'] = df_train['Fare'].mean()
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()

# log 적용 (편향된 데이터 보정)
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
df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.') 
    
df_test['Initial']=0
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
df_train.head(5)
df_all = pd.concat([df_train, df_test])
df_all.head(10)
df_all.tail(5)
df_all.reset_index(drop=True)
df_all.head()
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
# 정확도
print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
print(X_tr.shape)
print(X_vld.shape)
(prediction == y_vld).sum()/prediction.shape[0]
#pred_train_rf = model.predict(X_tr)
#pred_test_rf = model.predict(X_vld)
#score_train_rf = metrics.accuracy_score(pred_train_rf, y_tr)
#score_test_rf = metrics.accuracy_score(pred_test_rf, y_vld)
#print("RandomForest Train Score: ", score_train_rf)
#print("RandomForest Test Score: ", score_test_rf)
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
%%time

import xgboost as xgb

model_xgb = xgb.XGBClassifier(max_depth=9, learning_rate=0.01, n_estimators=500, reg_alpah=1.1,
                             colsample_bytree=0.9, subsample=0.9, n_jobs=5)
model_xgb.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=False, early_stopping_rounds=50)
pred_xgb = model_xgb.predict(X_vld)
score_xgb = metrics.accuracy_score(pred_xgb, y_vld)
print("XGBoost Test score: ", score_xgb)
from pandas import Series
feature_importance = model_xgb.feature_importances_
Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8, 8))
Series_feat_imp.sort_values(ascending=True).plot.barh()
plt.title("XGBoost")
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
%%time 

import lightgbm as lgbm

model_lgbm = lgbm.LGBMClassifier(max_depth=9, lambda_l1=0.1, lambda_l2=0.01, learning_rate=0.01,
                               n_estimators=500, reg_alpha=1.1, colsample_bytree=0.9, subsample=0.9, n_jobs=5)
model_lgbm.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=False, early_stopping_rounds=50,
              eval_metric="accuracy")
pred_lgbm = model_lgbm.predict(X_vld)
score_lgbm = metrics.accuracy_score(pred_lgbm, y_vld)
print("LightGBM Test Score: ", score_lgbm)

from pandas import Series
feature_importance = model_lgbm.feature_importances_
Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8, 8))
Series_feat_imp.sort_values(ascending=True).plot.barh()
plt.title("LightGBM")
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
!pip install catboost
%%time 

import catboost as cboost

model_cboost = cboost.CatBoostClassifier(depth=9, reg_lambda=0.1, learning_rate=0.01, iterations=500)
model_cboost.fit(X_tr, y_tr, eval_set=[(X_vld, y_vld)], verbose=False, early_stopping_rounds=50)
pred_cboost = model_cboost.predict(X_vld)
score_cboost = metrics.accuracy_score(pred_cboost, y_vld)
print("CatBoost Test Score: ", score_cboost)

from pandas import Series
feature_importance = model_cboost.feature_importances_
Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8, 8))
Series_feat_imp.sort_values(ascending=True).plot.barh()
plt.title("CatBoost")
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras import backend as K

model_mlp = Sequential()
model_mlp.add(Dense(45 ,activation='linear', input_dim=13))
model_mlp.add(BatchNormalization())

model_mlp.add(Dense(9,activation='linear'))
model_mlp.add(BatchNormalization())
model_mlp.add(Dropout(0.4))

model_mlp.add(Dense(5,activation='linear'))
model_mlp.add(BatchNormalization())
model_mlp.add(Dropout(0.2))

model_mlp.add(Dense(1,activation='relu', ))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
model_mlp.compile(optimizer=optimizer, 
            loss='binary_crossentropy', 
            metrics=['accuracy'])

hist = model_mlp.fit(X_tr, y_tr, epochs=500, batch_size=30, validation_data=(X_vld,y_vld), verbose=False)

pred_mlp = model_mlp.predict_classes(X_vld)[:,0]
score_mlp = metrics.accuracy_score(pred_mlp, y_vld)
print("MLP Test Score: ", score_mlp)

fig, loss_ax = plt.subplots(figsize=(10,10))

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
X_tr.shape

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras import backend as K

model_mlp = Sequential()
model_mlp.add(Dense(32 ,activation='relu', input_dim=13))  # X_tr.shape: (623, 13)
#model_mlp.add(BatchNormalization())

model_mlp.add(Dense(16,activation='relu'))
#model_mlp.add(BatchNormalization())
#model_mlp.add(Dropout(0.4))

model_mlp.add(Dense(8,activation='relu'))
#model_mlp.add(BatchNormalization())
#model_mlp.add(Dropout(0.2))

model_mlp.add(Dense(1,activation='sigmoid'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
model_mlp.compile(optimizer=optimizer, 
            loss='binary_crossentropy', 
            metrics=['accuracy'])

hist = model_mlp.fit(X_tr, y_tr, epochs=500, batch_size=30, validation_data=(X_vld,y_vld), verbose=False)

pred_mlp = model_mlp.predict_classes(X_vld)[:,0]
score_mlp = metrics.accuracy_score(pred_mlp, y_vld)
print("MLP Test Score: ", score_mlp)

fig, loss_ax = plt.subplots(figsize=(10,10))

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
WORK_DIR = './datasets/'
submission = pd.read_csv(WORK_DIR + 'sample_submission.csv')
submission.head()
# best model 선택
model = model_cboost

prediction = model.predict(X_test).astype('uint8')
submission['Survived'] = prediction

submission.to_csv('./titanic_best_model.csv', index=False)
type(prediction)
!head -10 ./titanic_best_model.csv
ROUND_NUM = 4
pred_rf = model.predict(X_vld)
score_rf = metrics.accuracy_score(pred_rf, y_vld)
score_rf = round(score_rf, ROUND_NUM)
print("RandomForest: ", score_rf)
pred_xgb = model_xgb.predict(X_vld)
score_xgb = metrics.accuracy_score(pred_xgb, y_vld)
score_xgb = round(score_xgb, ROUND_NUM)
print("XGBoost: ", score_xgb)
pred_lgbm = model_lgbm.predict(X_vld)
score_lgbm = metrics.accuracy_score(pred_lgbm, y_vld)
score_lgbm = round(score_lgbm, ROUND_NUM)
print("LGBM: ", score_lgbm)
pred_cboost = model_cboost.predict(X_vld)
score_cboost = metrics.accuracy_score(pred_cboost, y_vld)
score_cboost = round(score_cboost, ROUND_NUM)
print("CatBoost: ", score_cboost)
pred_mlp = model_mlp.predict_classes(X_vld)[:,0]
score_mlp = metrics.accuracy_score(pred_mlp, y_vld)
score_mlp = round(score_mlp, ROUND_NUM)
print("MLP: ", score_mlp)
df_score = pd.DataFrame({'model': ['RandomForest', 'XGBoost', 'LGBM', 'CatBoost', 'MLP'],
                        'score': [score_rf, score_xgb, score_lgbm, score_cboost, score_mlp]})
ax = df_score.plot.bar(x='model', y='score', rot=0, figsize=(10,5))
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
pred_avg = np.round(np.average([pred_rf, pred_xgb, pred_lgbm, pred_cboost, pred_mlp], axis=0)).astype(int)
score_avg = metrics.accuracy_score(pred_avg, y_vld)
score_avg = round(score_avg, ROUND_NUM)
print("AVG: ", score_avg)


pred_wavg = np.round(np.average([pred_rf, pred_xgb, pred_lgbm, pred_cboost, pred_mlp], 
                               weights=[0.2, 0.3, 0.2, 0.2, 0.1], axis=0)).astype(int)
score_wavg = metrics.accuracy_score(pred_wavg, y_vld)
score_wavg = round(score_wavg, ROUND_NUM)
print("Weighted Average: ", score_wavg)
# 최종 답안 제출: 테스트 데이터 사용
pred_rf = model.predict(X_test)
pred_xgb = model_xgb.predict(X_test)
pred_lgbm = model_lgbm.predict(X_test)
pred_cboost = model_cboost.predict(X_test)
pred_mlp = model_mlp.predict_classes(X_test)[:,0]
# 앙상블 
pred_wavg = np.round(np.average([pred_rf, pred_xgb, pred_lgbm, pred_cboost, pred_mlp],  
                                weights=[0.2, 0.3, 0.2, 0.2, 0.1], axis=0)).astype(int)

submission['Survived'] = pred_wavg
submission.to_csv('./titanic_wavg.csv', index=False)

from google.colab import files
files.download('./titanic_best_model.csv')

