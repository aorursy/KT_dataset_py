import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight') #matplotlib 내부적인 스킨 중 하나인 것 같다

import warnings

warnings.filterwarnings('ignore') #경고가 떴을시 처리하는 방식(무시하는)

%matplotlib inline 

#차트를 즉각적으로 출력해주는 명령어
data =  pd.read_csv('../input/titanic/train.csv')
data.head()
type(data)
data.columns
data.isnull()
data.isnull().sum()
f,ax = plt.subplots(1,2,figsize=(18,5)) 

#전체 차트모음의 배열(1행 2열)와 크기(가로길이, 세로길이)

data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', ax=ax[0],

                                        shadow=True)

# Survied 수치를 구하고 파이차트를 만든다

# 파라미터 상세 설명?

ax[0].set_title('Survived')

# 인덱스 0에 위치한 차트의 이름

ax[0].set_ylabel('')

# y축의 라벨을 비워두겠다는 의미

# 해당 함수가 없으면 제목이 y축으로 나온다

sns.countplot('Survived', data=data, ax=ax[1])

# Survied 의 빈도수를 재서 막대그래프 구현, 인덱스 1번에 위치.

ax[1].set_title('Survived')

# 인덱스 1번에 위치한 차트 이름

plt.show()
data.groupby(['Sex','Survived'])['Survived'].count()
data.groupby(['Sex','Survived']).size()
f,ax = plt.subplots(1,2,figsize=(18,8))

# figure

#   그림을 그릴 영역을 나타내는 객체, 위 파라미터로 설정

# axes

#   차트의 위치를 설정하는 변수 ex) axes[0] = df~ , axes[1] = df~

data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

#sns.barplot('Sex','Survived',data=data,ax=ax[0])

# 막대 그래프 비교 두가지 방법

#  data().groupby().mean().plot.bar()

#  sns.barplot()

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=data,ax=ax[1])

# hue : 막대 그래프(빈도)에서 분류 기준(집단형 변수)을 추가

ax[1].set_title('Sex : Survived vs Dead')

plt.show()
pd.crosstab(data.Pclass, data.Survived, margins=True)

# crosstab(행 데이터, 열 데이터, 합계 여부)
f,ax = plt.subplots(1,2,figsize=(18,8))

#차트를 그릴 영역을 설정

data['Pclass'].value_counts().plot.bar(ax=ax[0])

sns.countplot('Pclass',hue = 'Survived', data=data, ax=ax[1])
pd.crosstab([data.Sex, data.Survived], data.Pclass, margins=True)
sns.factorplot('Pclass','Survived',hue = 'Sex', data=data)

# 실선그래프 (X축, Y춧, hue추가기준)

# Y값을 Survied로 설정하여 Pclass별로 해당 평균값을 나타낸다(빈도 아님)
f,ax = plt.subplots(1,2,figsize=(18,8))

sns.violinplot('Pclass','Age',hue = 'Survived', data=data, split=True, ax=ax[0])

sns.violinplot('Sex','Age',hue = 'Survived', data=data, split=True, ax=ax[1])
data['Initial']=0

for i in data:

    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')
data.Initial
pd.crosstab(data.Sex, data.Initial)
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
pd.crosstab(data.Sex, data.Initial)
data.groupby('Initial')['Age'].mean()
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33

data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36

data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5

data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22

data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46
data.Age.isnull().sum()
f,ax = plt.subplots(1,2,figsize=(18,8))

data[data['Survived']==0].Age.plot.hist(ax=ax[0])

data[data['Survived']==1].Age.plot.hist(ax=ax[1])
sns.factorplot('Pclass','Survived',hue='Initial', data=data)

# hue 파라미터는 한 그래프에서 여러 실선으로 출력
sns.factorplot('Pclass','Survived',col='Initial', data=data)

# col 파라미터는 여러 그래프로 나누어 출력
pd.crosstab([data.Embarked,data.Pclass],[data.Sex,data.Survived],margins=True)

# 행, 열 파라미터에서 [] 기호를 활용하여 두개의 집단형 변수를 기준으로 둘 수 있다
sns.factorplot('Embarked','Survived',data=data)
f,ax = plt.subplots(2,2,figsize=(18,15))

sns.countplot('Embarked',data=data,ax=ax[0,0])

#data['Embarked'].value_counts().plot.bar(ax=ax[0,0])

#빈도수 그래프를 구현하는 두가지 방법

#  sns.countplot

#  df['column'].value_counts().plot

sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0,1])

sns.countplot('Embarked',hue='Survived',data=data,ax=ax[1,0])

sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1,1])
sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=data)

#파리미터 설명

# X축은 Pclass / Y축은 Survived / Sex별 실선 / Embarked별 그래프
data['Embarked'].fillna('S',inplace=True)
data['Embarked'].isnull().any()

#data['Embarked'].isnull().sum()

#null값의 유무를 확인하는 두가지 방법

#  isnull() + any / sum
pd.crosstab(data.SibSp, data.Survived)
f,ax = plt.subplots(1,2,figsize=(18,8))

sns.barplot('SibSp','Survived',data=data,ax=ax[0])

sns.factorplot('SibSp','Survived',data=data,ax=ax[1])
pd.crosstab(data.SibSp, data.Pclass)
pd.crosstab(data.Parch, data.Pclass)
f,ax = plt.subplots(1,2,figsize=(18,8))

sns.barplot('Parch','Survived',data=data,ax=ax[0])

sns.factorplot('Parch','Survived',data=data,ax=ax[1])
print(data['Fare'].max())

print(data['Fare'].min())

print(data['Fare'].mean())
f,ax = plt.subplots(1,3,figsize=(18,8))

sns.distplot(data[data['Pclass']==1].Fare,ax=ax[0])

sns.distplot(data[data['Pclass']==2].Fare,ax=ax[1])

sns.distplot(data[data['Pclass']==3].Fare,ax=ax[2])

# X축은 Fare변수값, Y축은 Fare.value_couts()
sns.heatmap(data.corr(),annot=True)

# annot는 각 셀마다 들어있는 상관계수

fig = plt.gcf()

fig.set_size_inches(10,8)
data['Age_band']=0

data.loc[data['Age']<=16,'Age_band']=0

data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1

data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2

data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3

data.loc[data['Age']>64,'Age_band']=4

data.head(2)
data['Age_band'].value_counts().to_frame()

#data.Age_band.value_counts().to_frame()

# 위아래 어느 형태든 상관없이 프레임화 가능
sns.factorplot('Age_band','Survived',data=data,col='Pclass')
data['Family_Size']=0

data['Family_Size']=data['Parch']+data['SibSp']#family size

data['Alone']=0

data.loc[data.Family_Size==0,'Alone']=1#Alone
f,ax = plt.subplots(1,2,figsize=(18,8))

sns.factorplot('Family_Size','Survived',data=data,ax=ax[0])

sns.factorplot('Alone','Survived',data=data,ax=ax[1])
sns.factorplot('Alone','Survived',data=data,col='Pclass',hue='Sex')
data['Fare_Range'] = pd.qcut(data['Fare'],4)

# pandas.qcut 수치를 N등분 하는 명령어

data.groupby(['Fare_Range'])['Survived'].mean().to_frame()
data['Fare_cat']=0

data.loc[data['Fare']<=7.91,'Fare_cat']=0

data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1

data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2

data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3
sns.factorplot('Fare_cat','Survived',data=data,hue='Sex')
data['Sex'].replace(['male','female'],[0,1],inplace=True)

data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],

                        [0,1,2,3,4],inplace=True)
data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)

sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})

fig=plt.gcf()

fig.set_size_inches(18,15)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.show()
df = data
df.columns
train_cols = df.columns[1:]

X = df[train_cols]

y = df['Survived']

y.value_counts()
from imblearn.under_sampling import RandomUnderSampler

X_sample, y_sample = RandomUnderSampler(random_state=0).fit_sample(X,y)

# 기존 데이터에서 랜덤으로 값을 추출(언더샘플링)
X_samp = pd.DataFrame(data=X_sample, columns=train_cols)

y_samp = pd.DataFrame(data=y_sample, columns=['Survived'])

# 추출한 값을 데이터프레인으로 변환

df_samp = pd.concat([X_samp,y_samp], axis=1)

# X,y 데이터프레임을 병합

df_samp['Survived'].value_counts()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,

                                                    stratify=y,random_state=0)

# stratify=y 설정하면 나누어진 데이터셋들도 0, 1을 각각 동일한 비율로 유지한채 분할된다

X_train.shape, X_test.shape, y_train.shape, y_test.shape
import statsmodels.api as sm

model = sm.Logit(y,X)

# 종속변수가 0,1 이므로 로지스틱 회귀분석

result = model.fit()

result.summary()
from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(random_state=10)

logit.fit(X_train,y_train)
print(logit.score(X_train, y_train))

print(logit.score(X_test, y_test))
X_train.boxplot()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = pd.DataFrame(scaler.transform(X_train))

X_test_scaled = pd.DataFrame(scaler.transform(X_test))
X_train_scaled.boxplot()
logit = LogisticRegression(random_state=10)

logit.fit(X_train_scaled, y_train)
score_tr = logit.score(X_train_scaled, y_train)

score_te = logit.score(X_test_scaled, y_test)

print(score_tr)

print(score_te)
# 설명값 수집 리스트

result_tr = []

result_te = []
result_tr.append(score_tr)

result_te.append(score_te)

print(result_tr)

print(result_te)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=10)

tree.fit(X_train_scaled, y_train)
print(tree.score(X_train_scaled, y_train))

print(tree.score(X_test_scaled, y_test))
tree = DecisionTreeClassifier(max_depth=3, random_state=10)

tree.fit(X_train_scaled, y_train)
print(tree.score(X_train_scaled, y_train))

print(tree.score(X_test_scaled, y_test))
score_tr = tree.score(X_train_scaled, y_train)

score_te = tree.score(X_test_scaled, y_test)
result_tr.append(score_tr)

result_te.append(score_te)

print(result_tr)

print(result_te)
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators= 100,

                               max_depth = 3, random_state=10)

forest.fit(X_train_scaled, y_train)
score_tr = forest.score(X_train_scaled, y_train)

score_te = forest.score(X_test_scaled, y_test)

print(score_tr)

print(score_te)
result_tr.append(score_tr)

result_te.append(score_te)

print(result_tr)

print(result_te)
data = pd.read_csv('../input/titanic/test.csv')



data['Initial']=0

for i in data:

    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')



# data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt',

#                          'Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other',

#                                        'Mr','Mr','Mr'],inplace=True)
pd.crosstab(data.Sex, data.Initial)
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33

data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36

data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5

data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22

data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46



data['Embarked'].fillna('S',inplace=True)



data['Age_band']=0

data.loc[data['Age']<=16,'Age_band']=0

data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1

data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2

data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3

data.loc[data['Age']>64,'Age_band']=4



data['Family_Size']=0

data['Family_Size']=data['Parch']+data['SibSp']#family size

data['Alone']=0

data.loc[data.Family_Size==0,'Alone']=1#Alone



data['Fare_Range'] = pd.qcut(data['Fare'],4)



data['Fare_cat']=0

data.loc[data['Fare']<=7.91,'Fare_cat']=0

data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1

data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2

data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3



data['Sex'].replace(['male','female'],[0,1],inplace=True)

data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],

                        [0,1,2,3,4],inplace=True)



data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
X_test = data.values

model = DecisionTreeClassifier(random_state=10)

model.fit(X_train_scaled, y_train)
prediction = model.predict(X_test)
submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission['Survived']=prediction
submission.to_csv('./my_first_submission.csv',index=False)