import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')

sns.set(font_scale = 2.5)



#missingno -- dataset에 채워지지 않은 null data 쉽게 보여줌

import missingno as msno

#ignore warnings

import warnings

warnings.filterwarnings('ignore')



#inline 하면 노트북에 바로 바로 볼 수 있음!

%matplotlib inline
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
df_train.head()
#통계치 반환 

df_train.describe()
df_test.describe()
#{:> 10} 오른쪽 정렬 {:<10} 왼쪽 정렬

#{:.2f} 결측치를 소수 두번째 자리까지 %로 나타내는거 

for col in df_train.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)
for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)
msno.matrix(df=df_train.iloc[:,:],figsize=(8,8),color =(0.8,0.5,0.2))
#missingno 데이터프레임 pandas indexing해주는거 원하는 row나 원하는 column 

#color에 있는 숫자는 rgb 숫자임

#figsize ㄹㅇ 사이즈 말하는거

msno.bar(df=df_test.iloc[:,:],figsize=(4,4),color =(0.3,0.2,0.2))
msno.matrix(df=df_test.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
#코드 함수 해석해보기 

#as=as[0] 도화지에 첫번째 그림

# plt.subplots(pyplot의 subplot은 팔레트 깐다고 생각.)

f, ax = plt.subplots(1,2, figsize =(17,8)) #1행2열 1행에 2개의 그림 figure size 가로로 18, 세로로 8

df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct ='%1.1f%%',

                                            ax=ax[0],shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived') #그래프 제목



plt.show()
plt.plot(df_train['Survived'].value_counts())
# count 객체가 몇 개임?

df_train[['Pclass','Survived']].groupby(['Pclass'],as_index =True).count()
#객체중 1인 애들 몇 개임?

df_train[['Pclass','Survived']].groupby(['Pclass'],as_index =True).sum()
df_train[['Pclass','Survived']].groupby(['Pclass'],as_index =True).mean()
#Pclass 행 Survived 열. gradient-- background_gradient 기울어지는 값에 따라 색 바꿔줄고임 ~~ 

pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).style.background_gradient(cmap='cool')
#ascend = False 내림차순 

df_train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar()
# firstclass에 있을떄 더 잘 사니까

# Pclass가 높을 수록 생존확률이 더 높다

# class를 쓰면 모델에 더 좋은 모델링?을 적용할 수 있음

y_position =1.02

f, ax = plt.subplots(1,2, figsize =(18,8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00',

                                                 '#FF3399'],ax =ax[0])

ax[0].set_title('Number of Passengers By Pclass',y=y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass',hue='Survived',data=df_train,ax=ax[1])

ax[1].set_title('Pclass: Survived vs Dead', y= y_position)

plt.show()
# female이 살 확률이 더 높다. 

f, ax = plt. subplots(1,2, figsize =(18,8))

df_train[['Sex','Survived']].groupby(['Sex'],as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=df_train,ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
df_train[['Sex','Survived']].groupby(['Sex'],as_index =False).mean().sort_values(by='Survived',ascending =False)
#pd의 crosstab == crosstable 이용 

pd.crosstab(df_train['Sex'],df_train['Survived'],margins =True).style.background_gradient(cmap='winter_r')
# 'Pclass' x축/'Survived' y축 

#  female이 살 확률이 더 높다! 좌석이 안좋아질수록 생존확률 낮아짐.. 

sns.factorplot('Pclass','Survived',hue='Sex',data=df_train,size = 6,aspect=1.5)
#축 ! 보는 방향을 바꾼거임. 

# col쓰면 pclass=1 ,2,3 별로 그래프 다 따로 볼 수 있음 ~

sns.factorplot(x='Sex',y='Survived',col='Pclass',data=df_train,satureation=.5,

              size = 9, aspect =1)
#

print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))

print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))

print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))

too = df_train['Age']
too.mean()
too.max(0)
too.min()
too.value_counts()
# kdeplot 데이터들의 분포가 어떻게 되어 있는지 대략 묘사/추정

# 히스토그램을 smooth하게 그린거 ㅋㅋ 

# pandas 에서 ['Survived'== 1] indexing하는 방법.

# df_train 안쪽 이해안감 ㅠㅠ

fig,ax = plt.subplots(1,1, figsize =(9,5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived ==1 ', 'Survived == 0'])

plt.show()
f = plt.figure(figsize =(10,10))

f.ax = plt.subplots(1,1,figsize =(10,10))

plt.figure (figsize = (10,10))
f = plt.figure(figsize = (5,5))

a = np.arange(100)

b = np.sin(a)

plt.plot(b)
plt.figure(figsize=(8,6))

df_train['Age'][df_train['Pclass']== 1].plot(kind='hist')

df_train['Age'][df_train['Pclass']== 2].plot(kind='hist')

df_train['Age'][df_train['Pclass']== 3].plot(kind='hist')

plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class','2nd Class','3rd Class'])
fig.ax = plt.subplots(1,1,figsize =(5,5))

a = np.arange(100)

b = np.sin(a)



ax.plot(b)
#전체 age데이터 셋에서 생존한 사람 분포. 

#생존확률에 따른 분포 볼 수 있음. 불린 이용

fig,ax = plt.subplots(1,1, figsize =(9,5))

sns.kdeplot(df_train[(df_train['Survived'] == 0)&(df_train['Pclass'] == 1)]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived'] == 0)&(df_train['Pclass'] == 1)]['Age'], ax=ax)

plt.legend(['Survived ==0 ', 'Survived == 1'])

plt.title('1st Class')

plt.show()
fig,ax = plt.subplots(1,1, figsize =(9,5))

sns.kdeplot(df_train[(df_train['Survived'] == 0)&(df_train['Pclass'] == 2)]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived'] == 1)&(df_train['Pclass'] == 2)]['Age'], ax=ax)

plt.legend(['Survived ==0 ', 'Survived == 1'])

plt.title('2nd Class')

plt.show()
fig,ax = plt.subplots(1,1, figsize =(9,5))

df_train['Age'][(df_train['Pclass'] == 1) & (df_train['Survived'] == 0)].plot(kind='hist')

df_train['Age'][(df_train['Pclass'] == 1) & (df_train['Survived'] == 1)].plot(kind='hist')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class','2nd Class','3rd Class'])
plt.figure(figsize=(8,6))

df_train['Age'][df_train['Pclass']== 1].plot(kind='hist')

df_train['Age'][df_train['Pclass']== 2].plot(kind='hist')

df_train['Age'][df_train['Pclass']== 3].plot(kind='hist')

plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class','2nd Class','3rd Class'])
# i가 1살부터 80살까지 변화시키는데 x축 : y축은 생존확률의 변화

# 나이가 어릴수록 생존확률 겁나 높은 것을 알 수 있음

# sum 머임.. 

cummulate_survival_ratio = []

for i in range(1, 80):

    cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))

    

plt.figure(figsize=(7, 7))

plt.plot(cummulate_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
#female 생존 확률이 더 높다 

f,ax = plt.subplots(1,2, figsize =(18,8))

sns.violinplot("Pclass","Age",hue="Survived",data=df_train,scale='count',split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age",hue = "Survived",data = df_train, scale='count',split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
#

f,ax = plt.subplots(1,1, figsize = (7,7))

df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar(ax=ax)
#survived 에 대해 정렬해주는 것을 확인 가능!

df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=True).mean().sort_values(by='Survived',ascending=True)
df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=True).mean().sort_index(by='Survived')
#여성이 많을수록 생존율이 높다

#C,Q에서 생존율이 높당

f,ax = plt.subplots(2,2, figsize=(20,15))

sns.countplot('Embarked', data = df_train, ax=ax[0,0])

ax[0,0].set_title('(1) No. Of Passsengers Boarded')

sns.countplot('Embarked', hue ='Sex', data=df_train, ax=ax[0,1])

ax[0,1].set_title('(2) Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Survived', data=df_train,ax=ax[1,0])

ax[1,0].set_title('(3) Embarked vs Survived')

sns.countplot('Embarked', hue = 'Pclass',data=df_train, ax=ax[1,1])

ax[1,1].set_title('(4) Embarked vs Pclass')

plt.subplots_adjust(wspace =0.2 , hspace =0.5) #plot간의 간격을 넣어준거!

plt.show()
df_train['FamilySize'] = df_train['SibSp']+df_train['Parch']

+1 #자신을 포함해야하니 1을 더한다

df_test['FamilySize']=df_test['SibSp']+df_test['Parch']+1 #자신을 포함해야하니 1을 더한다
df_train['SibSp']
print("Maximum size of Family: ",df_train['FamilySize'].max())

print("Minimum size of Family: ", df_train['FamilySize'].min())
#혼자 탄 사람은 500명, 가족단위로도 많이 탐

#familysize hue로 색깔 넣어줌. survived 한 사람들

f,ax = plt.subplots(1,3, figsize =(40,10))

sns.countplot('FamilySize',data=df_train,ax=ax[0])

ax[0].set_title('(1) No.Of Passengers Boarded', y=1.02)



sns.countplot('FamilySize',hue='Survived',data=df_train, ax=ax[1])

ax[1].set_title('(2) Survived countplot dependging on FamilySize', y=1.02)



df_train[['FamilySize','Survived']].groupby(['FamilySize'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize', y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
#이 데이터셋을 포함하는 내용

# 패턴 잘 못 보여줌ㅠ

fig,ax = plt.subplots(1,1, figsize=(8,8))

g = sns.distplot(df_train['Fare'],color = 'b',label ='Skewness : {:.2f}'.format(df_train['Fare'].skew()),ax =ax)

g = g.legend(loc ='best')
#log를 취해야 한다

df_test.loc[df_test.Fare.isnull(), 'Fare']=df_test['Fare'].mean() 



df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

# g로 반환 log를 취해서 scareness 줄여주고 --> feature engineering

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
#ticket도 뺄고임 

df_train['Ticket'].value_counts()
#age에 null data 몇개여?확인할 때

df_train['Age'].isnull().sum()
df_train['Age'].mean(0)
#row data 중 정규표현식에 맞는 애 추출해주는 method

df_train['Initial']=df_train['Name'].str.extract('([A-Za-z]+)\.')

df_test['Initial']=df_test['Name'].str.extract('([A-Za-z]+)\.')
pd.crosstab(df_train['Initial'],df_train['Sex']).T.style.background_gradient(cmap='summer_r')
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)



df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
df_train.groupby('Initial').mean()
df_train.groupby('Initial')['Survived'].mean().plot.bar()
## loc쓰면 null에 채워 넣을 수 있음. [행,열]

df_train.loc[(df_train['Age'].isnull()),:]
df_train['Age'].isnull()
# = 상수(위의 Mr인 사람의 agㄷ 평균)

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
df_train['Embarked'].isnull().sum()
df_train.shape
df_train['Embarked'].fillna('S',inplace=True)
df_train['Embarked'].isnull().sum()
df_train['Age_cat']=0
df_train.head()
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

    
df_train['Age_cat_2']=df_train['Age'].apply(category_age)
# all 모든게 true일때만 true

(df_train['Age_cat'] == df_train['Age_cat_2']).all()
# 하나라도 true면 any는 true! 

(df_train['Age_cat'] == df_train['Age_cat_2']).any()
# axis =1로 두면 축이 1

df_train.drop(['Age','Age_cat_2'],axis=1,inplace= True)

df_test.drop(['Age'],axis=1,inplace= True)
df_train.Initial.unique()
df_train.loc[df_train['Initial']=='Master','Initial']
df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
#특정 칼럼에 뭐 들어있는지 확인하고 싶을 때 사용법 2가지

df_train['Embarked'].unique()
df_train['Embarked'].value_counts()
# Embarked 숫자로 바꿈!

df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_train['Embarked'].isnull().any()
df_train['Sex'].unique()
df_train['Sex']=df_train['Sex'].map({'female':0, 'male':1})

df_test['Sex']=df_test['Sex'].map({'female':0, 'male':1})

heatmap_data = df_train[['Survived','Pclass','Sex','Fare','Embarked','FamilySize','Initial','Age_cat']]
# 불필요한 feature가 없누!선형관계가 1이면 둘중에 하나만 있어도 상관없다 이소리,,1 or -1이면 둘중에 하나는 필요없서

heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']] 



colormap = plt.cm.RdBu

plt.figure(figsize=(12,10))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,

           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})



del heatmap_data
# one-hot encoding 모델의 성능을 높이기 위해 원핫인코딩 ㅋㅋ 

df_train = pd.get_dummies(df_train, columns = ['Initial'],prefix = 'Initial')

df_test = pd.get_dummies(df_test, columns = ['Initial'],prefix = 'Initial')
df_train.head()
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')

df_train.head()
#필요한 columns만 남기고 다 지워버리기

df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_train.head()
df_test.head()
#sklearn은 사용해 머신러닝 모델 만들기

#importing all the required ML packages

from sklearn.ensemble import RandomForestClassifier #랜덤포레스트

from sklearn import metrics

from sklearn.model_selection import train_test_split
#학습 데이터와 target label를 분리하기 -- drop 사용해

X_train = df_train.drop('Survived',axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)
model = RandomForestClassifier()

model.fit(X_tr, y_tr)

prediction = model.predict(X_vld)
#모델 쩌러

print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
##feature importance 

##어떤 feature에 지금 만든 모델이 가장 많이 영향을 받았는 지 확인 가능
from pandas import Series



feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize = (8,8))

Series_feat_imp.sort_values(ascending = True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')

plt.show()
##prediction on test set

#생존 여부 예측해보기!
submission =  pd.read_csv('../input/titanic/gender_submission.csv')
submission.head()
prediction = model.predict(X_test)

submission['Survived']=prediction
submission.to_csv('/my_first_submission.csv',index=False)