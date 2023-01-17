import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

import warnings




plt.style.use('seaborn')

sns.set(font_scale=2.5)


warnings.filterwarnings('ignore')
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.describe()
df_test.describe()
for col in df_train.columns:

    msg = 'column: {:>12}\t Percent of NaN value : {:.2f}%'.format(

        col, 100*(df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)

for col in df_test.columns:

    msg = 'column : {:>12}\t Percent of NaN value : {:.2f}'.format(

    col, 100*(df_test[col].isnull().sum())/df_test[col].shape[0])

    print(msg)
msno.matrix(df = df_train.iloc[:,:], figsize=(8,8), color=(0.8, 0.5, 0.2))
msno.bar(df = df_train.iloc[:,:], figsize=(8,8), color=(0.8,0.5,0.2))
msno.bar(df = df_test.iloc[:,:], figsize=(8,8), color=(0.8,0.5,0.2))
f, ax = plt.subplots(1,2, figsize=(18,8))

df_train['Survived'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pipe plot - Survived')

ax[0].set_ylabel('')



sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived')



plt.show()

df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).count()
df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).sum()
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).mean()*100

df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
y_position = 1.02

f, ax = plt.subplots(1,2,figsize=(18,8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])

ax[0].set_title('Number of Passengers By Pclass', y=y_position)

ax[0].set_ylabel('Count')

ax[0].set_xlabel('Pclass')

sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()
f, ax = plt.subplots(1,2,figsize=(18,8))



df_train[['Sex','Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex',y=y_position)

sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead', y=y_position)

plt.show()
df_train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass','Survived',hue='Sex', data=df_train, size=6, aspect=1.5)
sns.factorplot(x='Sex',y='Survived', col='Pclass',data=df_train,size=9,aspect=1, satureation=.5)
print('the oldest passerger age : {:.1f} years'.format(df_train['Age'].max()))

print('the youngest passerger age : {:.1f} years'.format(df_train['Age'].min()))

print('the mean passerger age : {:.1f} years'.format(df_train['Age'].mean()))

f, ax = plt.subplots(1,1,figsize=(12,5))

#df_train[df_train['Survived']==1]['Age']

#backup 

# sns.kdeplot(df_train[df_train['Survived']==1]['Age'],ax=ax)

# sns.kdeplot(df_train[df_train['Survived']==0]['Age'],ax=ax)

# plt.legend(['Survived == 1','Survived == 0'])

# plt.show()
plt.figure(figsize=(10,6))

df_train['Age'][df_train['Pclass']==1].plot(kind='kde')

df_train['Age'][df_train['Pclass']==2].plot(kind='kde')

df_train['Age'][df_train['Pclass']==3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class','2nd Class', '3rd Class'])



cummulate_survival_ratio = []

for i in range(1,80):

    cummulate_survival_ratio.append(

        df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))
plt.figure(figsize=(7,7))

plt.plot(cummulate_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=y_position)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
f, ax = plt.subplots(1,2, figsize=(18,8))



sns.violinplot('Pclass', 'Age', hue='Survived', data=df_train,scale='count',split=True,ax=ax[0])

sns.violinplot('Sex', 'Age', hue='Survived', data=df_train, scale='count',split=True,ax=ax[1])



ax[0].set_title('Pclass and Age vs Survived')

ax[1].set_title('Sex and Age vs Survived')



ax[0].set_yticks(range(0,110,10))

ax[1].set_yticks(range(0,110,10))



plt.show()
f, ax = plt.subplots(1,1,figsize=(7,7))

mean_embarked = df_train[['Embarked','Survived']].groupby(['Embarked'],as_index=True).mean()

mean_embarked.sort_values(by='Survived',ascending=False).plot.bar(ax=ax)

plt.show()
mean_embarked
f, ax = plt.subplots(2,2, figsize=(20,15))



sns.countplot('Embarked', data=df_train, ax=ax[0,0])

ax[0,0].set_title('(1) No. of Passenger Boarded')

sns.countplot('Embarked', hue='Sex',data=df_train, ax=ax[0,1])

ax[0,1].set_title('(2) Male-Female Split for Embarked')

sns.countplot('Embarked', hue='Survived',data=df_train, ax=ax[1,0])

ax[1,0].set_title('(3) Embarked vs Survived')

sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1,1])

ax[1,1].set_title('(4) Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()



df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 #자신을 포함해야 하니 1을 더함.

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 #자신을 포함해야 하니 1을 더함.
print('Maximum size of Family : ', df_train['FamilySize'].max())

print('Minimum size of Family : ', df_train['FamilySize'].min())

f, ax = plt.subplots(1,3, figsize=(40,10))

sns.countplot('FamilySize', data=df_train, ax=ax[0])

ax[0].set_title('(1) No. of Passenger Boarded', y=y_position)



sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('(2) Survived countplot depending on FamilySize', y=y_position)



mean_Survived_FamilySize = df_train[['FamilySize','Survived']].groupby(['FamilySize'], as_index=True).mean()

mean_Survived_FamilySize.sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(2) Survived rate depending on Familysize', y=y_position)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
f, ax=plt.subplots(1,1,figsize=(8,8))

g = sns.distplot(

    df_train['Fare'],

    color='b',

    label='Skewness : {:.2f}'.format(df_train['Fare'].skew()),

    ax=ax)

g = g.legend(loc='best')
df_train['Fare'].skew()
df_test.loc[df_test.Fare.isnull(),'Fare'] = df_test['Fare'].mean()
df_train['Fare'] = df_train['Fare'].map(lambda i : np.log(i) if i > 0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i : np.log(i) if i > 0 else 0)
fig, ax = plt.subplots(1,1, figsize=(8,8))

g = sns.distplot(df_train['Fare'],color='b',label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
df_train.head()
df_train['Ticket'].value_counts()
df_train['Age'].isnull().sum()
df_train['Age'].isnull().value_counts()
df_train.Name
df_train.Name.str.extract('([A-Za-z]+)\.').head()
df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')

df_test['Initial'] = df_test.Name.str.extract('([A-Za-z]+)\.')
df_train.Initial.head()
df_test.Initial.head()
pd.crosstab(df_train['Initial'],df_train['Sex']).T

# .T에 의해서 세로로 정렬된 테이블이 가로 정렬로 변경됨.
pd.crosstab(df_train['Initial'],df_train['Sex']).T.style.background_gradient(cmap='summer_r')

#initial - sex 관계
df_train['Initial'].replace([

    'Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'

],[

    'Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mr'    

], inplace=True)



df_test['Initial'].replace([

    'Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'

],[

    'Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mr'    

], inplace=True)



# [리스트1],[리스트2] 의 형태로 적어주고, 리스트1의 위치기반 인자에 대해 리스트 2의 인자로 치환해줌
df_train.groupby("Initial").mean()
df_train.groupby("Initial")['Survived'].mean().plot.bar()
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

print('Embarked has ', sum(df_train['Embarked'].isnull()), ' Null values')
df_train['Embarked'].value_counts()
df_train['Embarked'].fillna('S',inplace=True)
df_train['Embarked'].isnull().sum()
df_train['Age'].describe()
df_train['Age_cat'] = 0

df_test['Age_cat'] = 0
df_train.loc[df_train['Age'] < 10, 'Age_cat'] = 0

df_train.loc[(df_train['Age'] >=10)&(df_train['Age'] < 20), 'Age_cat'] = 1

df_train.loc[(df_train['Age'] >=20)&(df_train['Age'] < 30), 'Age_cat'] = 2

df_train.loc[(df_train['Age'] >=30)&(df_train['Age'] < 40), 'Age_cat'] = 3

df_train.loc[(df_train['Age'] >=40)&(df_train['Age'] < 50), 'Age_cat'] = 4

df_train.loc[(df_train['Age'] >=50)&(df_train['Age'] < 60), 'Age_cat'] = 5

df_train.loc[(df_train['Age'] >=60)&(df_train['Age'] < 70), 'Age_cat'] = 6

df_train.loc[(df_train['Age'] >=70), 'Age_cat'] = 7
df_test
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

df_test['Age_cat'] = df_test['Age'].apply(category_age)
print('1번 방법, 2번방법 둘다 같은 결과를 내면 True를 줘야함 - > ',(df_train['Age_cat']==df_train['Age_cat_2']).all())
df_train.head()
df_test.head()
df_train.drop(['Age','Age_cat_2'],axis=1,inplace=True)

df_test.drop(['Age'],axis=1,inplace=True)
df_train['Initial'] = df_train['Initial'].map({

    'Master' : 0,

    'Miss' : 1,

    'Mr' : 2,

    'Mrs' : 3,

    'Other' : 4

})

df_test['Initial'] = df_test['Initial'].map({

    'Master' : 0,

    'Miss' : 1,

    'Mr' : 2,

    'Mrs' : 3,

    'Other' : 4

})
df_train['Embarked'].value_counts()
df_train['Embarked'].unique()
df_train['Embarked'] = df_train['Embarked'].map({

    'C' : 0,

    'Q' : 1,

    'S' : 2

})

df_test['Embarked'] = df_test['Embarked'].map({

    'C' : 0,

    'Q' : 1,

    'S' : 2

})
df_train['Embarked'].isnull().any()
df_train['Sex'] = df_train['Sex'].map({

    'female' : 0,

    'male' : 1

})

df_test['Sex'] = df_test['Sex'].map({

    'female' : 0,

    'male' : 1

})
heatmap_data = df_train[

    [

        'Survived',

        'Pclass',

        'Sex',

        'Fare',

        'Embarked',

        'FamilySize',

        'Initial',

        'Age_cat'

    ]

]



heatmap_data.astype(float).corr()

#default로 Pearson correlation 이 적용됨.
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Feature', y=1.05, size=20)

sns.heatmap(

    heatmap_data.astype(float).corr(),

    vmax=1.0, 

    linewidths=0.1,

    square=True,

    cmap=colormap,

    linecolor='white',

    annot=True,

    annot_kws={"size":20}

)



del heatmap_data
df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_train.head()
df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')

df_test.head()
df_train = pd.get_dummies(df_train,columns=['Embarked'],prefix='Embarked')
df_test = pd.get_dummies(df_test,columns=['Embarked'],prefix='Embarked')
df_train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1, inplace=True)

df_test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1, inplace=True)

df_train.head()
df_test.head()
#importing all the required ML packages

from sklearn.ensemble import RandomForestClassifier #유명한 randomforestclassfier입니다.

from sklearn import metrics # 모델의 평가를 위해서 사용한다.

from sklearn.model_selection import train_test_split # training set을 쉽게 나눠주는 함수 입니다.
# train 데이터프레임에서 survived 컬럼을 없애고 나머지 데이터를 가져감

X_train = df_train.drop('Survived',axis=1).values

# train 데이터프레임에서 survived 컬럼 데이터만 가져감

target_label = df_train['Survived'].values

# test 데이터프레임에서 모든 데이터를 가져옴

X_test = df_test.values

X_tr, X_vld, y_tr, y_vld = train_test_split(

    X_train, 

    target_label, 

    test_size=0.3, 

    random_state=2018

)
model = RandomForestClassifier()

model.fit(X_tr,y_tr)

prediction = model.predict(X_vld)
print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(

    y_vld.shape[0],

    100*metrics.accuracy_score(prediction,y_vld)

)

     )
X_vld.shape

#shape를 하면, (행,열) 형태로 숫자를 튜플로 반화해줌
from pandas import Series



feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8,8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature Importance')

plt.ylabel('Feature')

plt.show()
submission = pd.read_csv('../input/gender_submission.csv')
submission.head()
# 테스트 데이터를 가지고 결과값을 예측하고, 먼저 사용했던 변수 prediction에 덮어 씌움

# 그리고, 결과용 csv파일의 컬럼 Survived에 예측한 값을 다시 한 번 덮어 씌워서 결과제출을 준비함.

# 여기서 주의할 점은, 테스트 데이터와 제출용데이터의 row 숫자(데이터 수)가 같은지 체크 할 것.

prediction = model.predict(X_test)

submission['Survived'] = prediction
submission.head()
submission.to_csv('./Titanic_simple_submission.csv', index=False)