import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno #data set에 포함되지 않은 NULL 값을 표현해줌

import warnings #ignore warning

%matplotlib inline



plt.style.use('seaborn') #차트 스타일을 seaborn 으로 사용

sns.set(font_scale=2.5) #차트 글자를 2.5로 고정

warnings.filterwarnings('ignore') # warning 무시

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head(10) #default = 5, data set을 원하는 개수별로 나타냄
df_train.describe() #data set에 평균값을 나타내줌
df_train.columns
for col in df_train.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100*(df_train[col].isnull().sum())/df_train[col].shape[0])

    print(msg)

'''

각 col에 NaN data의 Percent를 보기위한 작업

즉 총 891개의 data set에서 각 columns들이 차지하는 NaN 비율을 계산

'''
for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100*(df_test[col].isnull().sum())/df_test[col].shape[0])

    print(msg)
msno.matrix(df=df_train.iloc[:,:],figsize=(8,8),color=(0.5,0.5,0.5))

'''

msno는 아래 그림과 같은 matrix를 만들어준다.

빈칸이 의미하는것은 NaN값이다.

'''
msno.bar(df=df_train.iloc[:,:],figsize=(8,8),color=(0.5,0.5,0.5))

#matrix 말고 bar로 NaN 값을 표현
f, ax = plt.subplots(1,2,figsize=(18, 8))

#plt.subplots(low,col,figsize=(low,col)) -> 1행에 2개의 그림을 그리는것을 의미한다, figsize는 그림의 크기를 조절한다.



df_train['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct= '%1.1f%%',ax=ax[0],shadow=True)

# df_train['Survived'].value_counts() = 사망,생존을 0,1값으로 count해준다. 이것의 data type은 series이므로 plot을 갖는다. 즉 연속적인 값이므로 표현 할 수 있다.

# explode는 pie chart 간격을 주는것, autopct는 pie chart에 percent표시, ax는 열을 의미하므로 2개가 존재하고 첫번째 열에 pie chart를 그려주는것이다.

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=df_train, ax=ax[1])

#countplot('col name', data = data set, ax=position)

ax[1].set_title('Count plot - Survived')

plt.show()
#Pclass 에 따른 생존률을 알아보자

df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).count()
df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).sum()
df_train[['Pclass','Survived']]

#원하는 col의 값을 확인하고싶을때 의 문법
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')

#crosstab을 이용하여 Survived의 값에 따른 Pclass를 표현
df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending = False).plot()

# Survived값에 따른 Pclass의 평균값을 sort_values로 정렬, ascending = False(내림차순),True(오름차순), 코드형식이 pandas DataFrame이기 때문에 plot 을 가지고있다 -> bar로도 표현가능
y_position = 1.02

f, ax = plt.subplots(1, 2, figsize=(18,8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])

ax[0].set_title('Number of passengers By Pclass', y=y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass',hue = 'Survived', data=df_train, ax=ax[1])

# hue를 이용하여 Survived를 색깔로 구분하여 보여준다

ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()
f, ax = plt.subplots(1, 2, figsize=(18,8))

df_train[['Sex','Survived']].groupby(['Sex'],as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived', data =df_train, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
df_train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by = 'Survived', ascending = False)
pd.crosstab(df_train['Sex'], df_train['Survived'],margins = True ).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass','Survived', hue='Sex', data=df_train, size=5 , aspect=2)

#축을 정해주지 않고 그래프를 나타내는 방법
sns.factorplot(x='Sex',y='Survived',col='Pclass',data=df_train, saturation=5, size=9, aspect=1)

#축을 정해주고 그래프를 나타내는 방법, hue 대신 col을 사용했을때
sns.factorplot(x='Sex',y='Survived',hue='Pclass',data=df_train, saturation=5, size=9, aspect=1)

#hue 를 사용했을때
print('제일 나이 많은 탑승객: {:.1f} years'.format(df_train['Age'].max()))

print('제일 나이 적은 탑승객: {:.1f} years'.format(df_train['Age'].min()))

print('탑승객 평균 나이: {:.1f} years'.format(df_train['Age'].mean()))

fig, ax = plt.subplots(1, 1, figsize=(9,5))

sns.kdeplot(df_train[df_train['Survived']==1]['Age'], ax=ax)

plt.title('Survived')

sns.kdeplot(df_train[df_train['Survived']==0]['Age'], ax=ax)

plt.legend(['Survived=1','Survived=0'])

plt.show()





'''

kdeplot는 히스토그램을 이어서 그려주는것 

df_train[df_train['Survived']==1

df_train에 Survived중 1값인것만 추려냄

df_train[df_train['Survived']==0

df_train에 Survived중 0값인것만 추려냄

'''
plt.figure(figsize=(8, 6))

df_train['Age'][df_train['Pclass']==1].plot(kind='kde')

df_train['Age'][df_train['Pclass']==2].plot(kind='kde')

df_train['Age'][df_train['Pclass']==3].plot(kind='kde')

plt.legend(['1st Pclass','2nd Pclass','3nd Pclass'])

plt.xlabel('Age')

plt.title('Age Distribution within classes')
fig, ax = plt.subplots(1, 1, figsize=(9,5))

sns.kdeplot(df_train[(df_train['Survived']==0)& (df_train['Pclass']==1)]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived']==1)& (df_train['Pclass']==1)]['Age'], ax=ax)

plt.title('1st Class')

plt.legend(['Survived=1','Survived=0'])
fig, ax = plt.subplots(1, 1, figsize=(9,5))

sns.kdeplot(df_train[(df_train['Survived']==0)& (df_train['Pclass']==2)]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived']==1)& (df_train['Pclass']==2)]['Age'], ax=ax)

plt.title('2nd Class')

plt.legend(['Survived=1','Survived=0'])
fig, ax = plt.subplots(1, 1, figsize=(9,5))

sns.kdeplot(df_train[(df_train['Survived']==0)& (df_train['Pclass']==3)]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived']==1)& (df_train['Pclass']==3)]['Age'], ax=ax)

plt.title('3nd Class')

plt.legend(['Survived=1','Survived=0'])
change_age_survival_ratio = []



for i in range(1,80):

    change_age_survival_ratio.append(df_train[df_train['Age']<i]['Survived'].sum()/len(df_train[df_train['Age']<i]['Survived']))

    

plt.figure(figsize=(7,7))

plt.plot(change_age_survival_ratio)

plt.title('Suvival rate change depending on range of Age', y=1)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
i=10

df_train[df_train['Age']<i]['Survived'].sum()/len(df_train[df_train['Age']<i]['Survived'])
df_train[df_train['Age']<10]['Survived'].sum()
len(df_train[df_train['Age']<10]['Survived'])
f, ax = plt.subplots(1,2,figsize=(18,8))

sns.violinplot('Pclass','Age',hue='Survived',data=df_train,scale='count',split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,20))



sns.violinplot('Sex','Age',hue='Survived',data=df_train, scale='count',split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,20))

plt.show()
f, ax = plt.subplots(1,1,figsize=(7,7))

df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar(ax=ax)
df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=True).mean().sort_values(by='Embarked')
f, ax = plt.subplots(2, 2, figsize=(20,15))

sns.countplot('Embarked', data=df_train, ax=ax[0,0])

ax[0,0].set_title('(1) No. Of Passengers Boared')



sns.countplot('Embarked',hue='Sex',data=df_train,ax=ax[0,1])

ax[0,1].set_title('(2) Male-Female split for embarked')



sns.countplot('Embarked',hue='Survived',data=df_train,ax=ax[1,0])

ax[1,0].set_title('(3) Embarked vs Survived')



sns.countplot('Embarked',hue='Pclass',data=df_train,ax=ax[1,1])

ax[1,1].set_title('(4) Embarked vs Pclass')



plt.subplots_adjust(wspace=0.2, hspace=0.5)

# 그래프 간격을 조절함

plt.show()

df_train['FamilySize'] = df_train['SibSp']+df_train['Parch']+1

df_test['FamilySize'] = df_test['SibSp']+df_test['Parch']+1

#가족 프레임을 생성해준다. { SibSP(형제,자매,배우자) + Parch(부모,자식) }+자기자신(1)
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()
f, ax = plt.subplots(1,3,figsize=(40,10))

sns.countplot('FamilySize', data=df_train,ax=ax[0])

ax[0].set_title('(1)No. Of Passenger Boarded',y=1.02)



sns.countplot('FamilySize',hue='Survived', data=df_train,ax=ax[1])

ax[1].set_title('(2) Survived countplot depending on FamilySize', y=1.02)



df_train[['FamilySize','Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived countplot depending on FamilySize', y=1.02)



plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()
fig, ax = plt.subplots(1,1,figsize=(8,8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness:{:.2f}'.format(df_train['Fare'].skew()),ax=ax)

g = g.legend(loc='best')



'''

Skewness = "왜도" 라는뜻으로 통계학에서 자료의 분포모양이 평균을 중심으로부터 한 쪽으로 치우쳐져있는 경향을 나타내는 척도 이다.

Skewness값은 좌측으로 쏠리면 양수, 우측으로 쏠리면 음수로 나타난다. 

distplot series값을 히스토그램으로 나타내줌



아래 분포도로 학습을 하면 제대로된 학습결과가 나오지 않을수도 있기때문에 Log값을 취해준다.

'''

df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i>0 else 0)



# lambda 문법은 한줄로 코딩을 할 수 있는 파이썬의 특징을 갖는 문법이다. map을 이용하여 값 치환을 해줌, 주로 map은 lambda와 많이 쓰인다. 

# i값이 0 이상이면 log(i)를 취하고, 아니라면 0값을 취한다.
fig, ax = plt.subplots(1,1,figsize=(8,8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness:{:.2f}'.format(df_train['Fare'].skew()),ax=ax)

g = g.legend(loc='best')



'''

Log 값을 적용시켜서 Skewness값이 0에 수렴하도록 만들어줄수 있다. -> 학습에 효과적인 데이터값

'''
df_train['Initial']=df_train['Name'].str.extract('([a-zA-Z]*)\.')

df_test['Initial'] = df_test['Name'].str.extract('([a-zA-Z]*)\.')

# [a-zA-Z]* : 알파벳 모두(소문자, 대문자 상관없이) *(무한반복) \.(.이 붙은 것들 전부 추출)

pd.crosstab(df_train['Initial'],df_train['Sex']).T.style.background_gradient(cmap='summer_r')
df_train['Initial'].replace(['Mile','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],

                            ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mr'], inplace=True)



df_test['Initial'].replace(['Mile','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],

                            ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mr'], inplace=True)
df_train.groupby('Initial').mean()
df_train.groupby('Initial')['Survived'].mean().plot.bar()
df_all = pd.concat([df_train, df_test])

# concat을 이용하여 df_train과 df_test를 합쳤다.
df_all.groupby('Initial').mean()
df_train.loc[2:5,:]

# location 을 활용하여 원하는 범위의 값만 불러올수 있다
df_train.loc[df_train['Survived'] == 1]

# location을 이용하여 Survived 값이 1인 데이터만 불러온다 -> bolean Indexing
df_train.loc[df_train['Age'].isnull() & (df_train['Initial']=='Mr')]

# location을 이용하여 Age Frame에 Null값이고 Initial에 Mr 이 포함된 사람만 추출한다.
df_train.loc[df_train['Age'].isnull() & (df_train['Initial']=='Mr'), 'Age'] =33

# 나이값이 Null이고 이름에 Mr이 붙은사람의 나이를 33세로 지정해준다.

df_train.loc[(df_train['Initial']=='Mr'), 'Age']

# Age에 Null 값이 없으므로 이름이 Mr인 사람의 나이만 확인한다
df_train.loc[df_train['Age'].isnull() & (df_train['Initial']=='Mr'), 'Age'] =33

df_train.loc[df_train['Age'].isnull() & (df_train['Initial']=='Mrs'), 'Age'] =20

df_train.loc[df_train['Age'].isnull() & (df_train['Initial']=='Other'), 'Age'] =5

df_train.loc[df_train['Age'].isnull() & (df_train['Initial']=='Miss'), 'Age'] =48

df_train.loc[df_train['Age'].isnull() & (df_train['Initial']=='Master'), 'Age'] =27





df_test.loc[df_test['Age'].isnull() & (df_test['Initial']=='Mr'), 'Age'] =33

df_test.loc[df_test['Age'].isnull() & (df_test['Initial']=='Mrs'), 'Age'] =20

df_test.loc[df_test['Age'].isnull() & (df_test['Initial']=='Other'), 'Age'] =5

df_test.loc[df_test['Age'].isnull() & (df_test['Initial']=='Miss'), 'Age'] =48

df_test.loc[df_test['Age'].isnull() & (df_test['Initial']=='Master'), 'Age'] =27



print('train Null data percent = {:.0f}%'.format(df_train['Age'].isnull().sum()))

print('test Null data percent = {:.0f}%'.format(df_test['Age'].isnull().sum()))



# df_train과 df_test 의 Age frame에 Null값을 전부 채워 Null값을 없엠
df_train['Embarked'].isnull().sum()

df_train['Embarked'].fillna('S',inplace=True)

# Embarked frame에 Null값이 2개뿐이므로 가장많은 S로 값을 채워주고 마무리
df_train['Age_cat'] = 0
df_train.loc[df_train['Age']<10,'Age_cat'] = 0

df_train.loc[(10<= df_train['Age']) & (df_train['Age']<20) ,'Age_cat'] =1

df_train.loc[(20<= df_train['Age']) & (df_train['Age']<30) ,'Age_cat'] =2

df_train.loc[(30<= df_train['Age']) & (df_train['Age']<40) ,'Age_cat'] =3

df_train.loc[(40<= df_train['Age']) & (df_train['Age']<50) ,'Age_cat'] =4

df_train.loc[(50<= df_train['Age']) & (df_train['Age']<60) ,'Age_cat'] =5

df_train.loc[(60<= df_train['Age']) & (df_train['Age']<70) ,'Age_cat'] =6

df_train.loc[(70<= df_train['Age']),'Age_cat'] =7
df_test['Age_cat'] = 0

df_test.loc[df_test['Age']<10,'Age_cat'] = 0

df_test.loc[(10<= df_test['Age']) & (df_test['Age']<20) ,'Age_cat'] =1

df_test.loc[(20<= df_test['Age']) & (df_test['Age']<30) ,'Age_cat'] =2

df_test.loc[(30<= df_test['Age']) & (df_test['Age']<40) ,'Age_cat'] =3

df_test.loc[(40<= df_test['Age']) & (df_test['Age']<50) ,'Age_cat'] =4

df_test.loc[(50<= df_test['Age']) & (df_test['Age']<60) ,'Age_cat'] =5

df_test.loc[(60<= df_test['Age']) & (df_test['Age']<70) ,'Age_cat'] =6

df_test.loc[(70<= df_test['Age']),'Age_cat'] =7



# 이와같은 방법으로 Categorize를 할 수 있다.
df_train.head(10)
def category_age(x):

    if x <10:

        return 0

    elif x<20:

        return 1

    elif x<30:

        return 2

    elif x<40:

        return 3

    elif x<50:

        return 4

    elif x<60:

        return 5

    elif x<70:

        return 6

    else:

        return 7
#df_train['Age'].apply(category_age)

# apply는 함수를 이용하여 적용해줄때 쓰임
df_train['Age_cat_2'] = df_train['Age'].apply(category_age)

# 함수와 apply를 이용하여 이용하여 위의 방법보다 좀더 쉽게 Categorize를 할 수 있다.
print((df_train['Age_cat'] == df_train['Age_cat_2']).all())

'''

all을 쓸경우 Data 값이 모두 True일때 True를 반환, 아니면 False

any를 쓸경우 Data 값이 하나라도 True이면 True를 반환, 아니면 False



'''
df_train.drop(['Age','Age_cat_2'], axis=1, inplace=True)

df_test.drop(['Age'],axis=1,inplace=True)

# 필요없는 col을 없에기위해 drop을 사용 
df_train.Initial.unique()
df_train['Initial']=df_train['Initial'].map({'Master':0,'Miss':1,'Mr':2,'Mrs':3,'Other':4})

df_test['Initial']=df_test['Initial'].map({'Master':0,'Miss':1,'Mr':2,'Mrs':3,'Other':4})
df_train.Embarked.unique()
df_train['Embarked'].value_counts()
df_train['Embarked'] = df_train['Embarked'].map({'C':0,'Q':1,'S':2})

df_test['Embarked'] = df_test['Embarked'].map({'C':0,'Q':1,'S':2})
df_train['Embarked'].isnull().any()

#Embarked의 Null값을 모두 채웠기때문에 False값이 나옴
df_train['Sex']=df_train['Sex'].map({'female':0,'male':1})

df_test['Sex']=df_test['Sex'].map({'female':0,'male':1})
heatmap_data = df_train[['Survived','Pclass','Sex','Fare','Embarked','FamilySize','Initial','Age_cat']]
colormap = plt.cm.BuGn

plt.figure(figsize=(12,10))

plt.title('Pearson Correlation of Features', y=1.05,size=20)

sns.heatmap(heatmap_data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,linecolor='White',annot=True,annot_kws={'size':16})

del heatmap_data

'''

astype은 heatmap data의 데이터를 float형태로 바꿔주는것

corr = Correlation (상관관계)를 구해줌

''' 
df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')



'''

pandas를 이용하여 One-Hot encoding을 매우 간단하게 할 수 있다.

pd.get_dummies를 이용하면 One-Hot encoding이 되는데 이때 One-Hot encoding을 한 카테고리의 수대로 column이 생기게 된다.

새로 생긴 column들을 prefix를 이용하여 구분하기 쉽게 만들어줄수있다.



하지만 dummies의 단점은 카테고리가 100개 정도 되는 data set일 경우 학습하는데 매우 큰 영향을 끼칠수 있으므로 주의해야한다.

'''

df_train.head()
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')

#위와 같은 설명이다.
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 필요한 columns만 남기고 모두 지운다
df_train.head()
df_test.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics # 모델의 평가를 위해서 씁니다

from sklearn.model_selection import train_test_split # traning set을 쉽게 나눠주는 함수입니다.
X_train = df_train.drop('Survived', axis = 1).values

target_label = df_train['Survived'].values

X_test = df_test.values
(X_test== np.inf).sum()
(X_test== -np.inf).sum()
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)

# supervised learning, test_size = 0.3 ->예측 데이터 30%만 사용 , train_size = 70% ->훈련 데이터 70% 사용
model = RandomForestClassifier()

model.fit(X_tr,y_tr)

# RandomForestClassifier default setting 으로 학습
prediction = model.predict(X_vld)

#학습한 결과로 예측
print('총 {}명 중 {:.2f}% 정확도로 생존 맞춤'.format(y_vld.shape[0],100*metrics.accuracy_score(prediction,y_vld)))

#metrics.accuracy_score(prediction,y_vld) 에서 prediction과 y_vld를 비교하여 정확도를 뽑아냄
from pandas import Series
feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)
feature_importance
df_test.columns
df_train.head()
df_test.head()
plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')

plt.show()
submission = pd.read_csv('../input/gender_submission.csv')
submission.head(10)
prediction = model.predict(X_test)

submission['Survived'] = prediction
submission.to_csv('./my_first_submission.csv', index=False)
for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100*(df_test[col].isnull().sum())/df_test[col].shape[0])

    print(msg)
df_test['Fare'].mean()
df_test['Fare'].fillna(35, inplace=True)
#찾는방법

#(df_train['feature']== np.inf).sum()

#(df_train['feature']== -np.inf).sum()
#채우는법

#df_train[feat_name].replace([np.inf, -np.inf], 0, inplace=True)