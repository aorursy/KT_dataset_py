import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn 라이브러리 세팅
plt.style.use('seaborn')   # matplot 기본 그림 말고 seaborn 그림 스타일 사용
sns.set(font_scale=2.5)    # 폰트 사이즈 2.5로 고정

# null 데이터를 시각화하여 보여주는 라이브러리
import missingno as msno   

# 오류 무시하는 코드 
import warnings
warnings.filterwarnings('ignore')

# matplot 라이브러리 사용해 시각화한 뒤 show했을 때 새로운 창이 아닌 노트북에서 바로 확인 가능하도록
%matplotlib inline
df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
df_train.head()
# target label은 우리가 예측하려는 survived
# feature는 survived 제외한 나머지
df_train.shape
df_train.info()

# pclass: ordinal (순서가 있는 카테고리형)
# sex: binary형이므로 나중에 수치형으로 바꿔줘야 함
# age: 연속적인 수치형
# ticket: 문자형
# fare: 연속적인 수치형
# embarked: 순서가 없는 카테고리형

# 카테고리형은 원 핫 인코딩으로 데이터 처리 필요
df_train.describe()
# passengerid count는 행 개수인 891과 일치하는데 age는 714개임 -> null데이터가 존재한다는 것
df_test.head()
df_test.info()
df_test.describe()
df_train.isnull().sum() 
df_test.isnull().sum()
# matrix는 null값이 어디에 얼마나 촘촘하게 위치하는지를 알 수 있음
msno.matrix(df=df_train.iloc[:, :], figsize = (8,8), color=(0.8, 0.5, 0.2))
# bar는 null의 %를 알 수 있음
msno.bar(df=df_train.iloc[:, :], figsize = (8,8), color=(0.8, 0.5, 0.2))
# fig: 전체 틀
# ax: 전체 틀안에 그려질 그래프 개수
fig, ax = plt.subplots(1, 2, figsize=(18, 8)) # 1줄에 2개 그래프 그림

# 첫번째 그래프 
df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
# df_train['Survived'].value_counts()는 시리즈타입이라 .plot하면 matplot과 연결됨
# explode=[0, 0.1]: 파이 조각 돌출크기로 0하면 돌출되지 않음. 0.1하면 돌출되어 튀어나옴
# autopct: 백분율로 %1.1f%%하면 소수점 1자리까지 %로 표기
# ax=ax[0]: 두가지 위치 중 첫번째 위치

# 첫번째 그래프 설정
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('') # y축 레이블 없앰

# 두번째 그래프
# countplot은 데이터프레임으로만 사용가능
sns.countplot('Survived', data=df_train, ax=ax[1])  # (x축에는 열, 사용할 데이터, 두번째 위치)

# 두번째 그래프 설정
ax[1].set_title('Count plot - Survived')

plt.show()

# 사망자 0이 61%, 생존자 1이 38.4% -> 타겟 레이블의 분포가 제법 균일함
df_train[['Pclass', 'Survived']].groupby(['Pclass']).count() 
# count는 행 개수를 세므로 각 클래스의 인원을 알 수 있음 (칼럼이름이 survived이지 생존자가 아님)
df_train[['Pclass', 'Survived']].groupby(['Pclass']).sum()
# sum을 하면 생존자인 1을 모두 더하므로 1등급 인원 중에 몇 명이 살았는지 알 수 있음 (216명 중 136명 생존)
df_train[['Pclass', 'Survived']].groupby(['Pclass']).mean()
# mean을 하면 각 클래스별 생존율을 구할 수 있음
# margins=True는 합계인 All 표시
# .style.background_gradient(cmap='summer_r'): 숫자 크기에 따른 색깔표시(cmap은 컬러맵)   
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')  
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()  
# as_index=True하면 pclass를 인덱스로 두므로 pclass와 survived 칼럼 모두 시각화 됨
# 클래스가 높을수록 생존율이 높은 것을 알 수 있음
y_position = 1.02
fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# 첫번째 그래프
df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])

# 제목, y축 설정
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')

# 두번째 그래프
sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1]) # hue='Survived'하면 생존,사망별로 색깔 나눔

# 제목 설정
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()

# 왼쪽 그래프는 클래스별 승객수로 3등급 승객이 가장 많음
# 오른쪽 그래프에서 각 클래스별 생존자를 보면 티켓 등급이 높을수록 생존자가 많음
# 클래스에 따라 생존율이 달라진다는 것을 확인했으므로 pclass피처는 모델에 사용하기 좋은 칼럼임을 확인
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending = False)
pd.crosstab(df_train['Sex'], df_train['Survived'], margins = True).style.background_gradient(cmap = 'summer_r')
fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# 첫번째 그래프
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

# 제목 설정
ax[0].set_title('Survived vs Sex')

# 두번째 그래프
sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

# 제목 설정
ax[1].set_title('Sex: Survived vs Dead')

plt.show()

# 남자보다 여자의 생존율이 더 높음
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)  # pclass x축, survived y축, aspect는 높이를 일정하게 유지하면서 너비 변경 
# 모든 클래스에서 남자보다 여자가 살 확률이 높음
# 또한 남녀 상관없이 클래스가 높으면 살 확률이 높음
sns.factorplot(x='Sex', y='Survived', col='Pclass', data=df_train, satureation=.5, size=9, aspect=1)
print(df_train['Age'].max())  # 최고 나이
print(df_train['Age'].min())  # 최저 나이
print(df_train['Age'].mean()) # 평균 나이
df_train[df_train['Survived'] == 1]['Age']  # 생존자인 사람의 나이 출력
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

# KDE
sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

# 레이블 설정
plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()

# 생존자에 나이어린 승객이 꽤 있음을 알 수 있음
plt.figure(figsize=(8, 6))

# KDE
df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')
df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

# 설정
plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])

# 클래스가 높을수록 나이 많은 사람의 비중이 커짐
fig, ax = plt.subplots(1,2,figsize=(18,8))

# 첫번째 그래프
sns.violinplot("Pclass", "Age", hue = "Survived", data = df_train, sclae = 'count', split=True, ax=ax[0])

# 설정
ax[0].set_title("Pclass and Age vs Survived")
ax[0].set_yticks(range(0,110,10))

# 두번째 그래프
sns.violinplot("Sex", "Age", hue="Survived", data = df_train, scale='count', split=True,ax=ax[1])

# 설정
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))

plt.show()

# 왼쪽 그래프는 등급별 나이 분포와 그에 따른 생존여부를 나타냄
# 오른쪽 그래프는 성별별 나이 분포와 그에 따른 생존여부를 나타냄
# 모든 클래스에서 나이가 어릴수록 생존자가 많고, 여성이 많이 생존한 것을 알 수 있음
fig, ax = plt.subplots(1,1,figsize=(7,7))

df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending = False).plot.bar(ax=ax)

# c에서 탑승한 사람의 생존율이 높음
f,ax=plt.subplots(2, 2, figsize=(15,10))

# (1)
sns.countplot('Embarked', data=df_train, ax=ax[0,0])
ax[0,0].set_title('(1) No. Of Passangers Boarded')

# (2)
sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax[0,1])
ax[0,1].set_title('(2) Male-Female Split for Embarked')

# (3)
sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1,0])
ax[1,0].set_title('(3) Embarked vs Survived')

# (4)
sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1,1])
ax[1,1].set_title('(4) Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2, hspace = 0.5)
plt.show()

# (1) S 에서 가장 많은 사람이 탑승
# (2) C와 Q 는 남녀의 비율이 비슷하고, S는 남자가 더 많음
# (3) S인 경우 생존율이 낮음
# (4) pclass를 하니 C가 생존율 높은 이유는 클래스가 높은 사람이 많이 탔기 때문. S는 3등급이 많아서 생존율이 낮음
df_train['Family']=df_train['SibSp'] + df_train['Parch'] + 1
df_test['Family']=df_test['SibSp'] + df_test['Parch'] + 1
# 자기 자신을 포함해야 하니 1을 더함
print(df_train['Family'].max())  # 최대 가족 수
print(df_train['Family'].min())  # 최저 가족 수
f,ax=plt.subplots(1, 3, figsize=(40,10))

# (1)
sns.countplot('Family', data=df_train, ax=ax[0])
ax[0].set_title('(1) No. Of Passangers Boarded', y=1.02)

# (2)
sns.countplot('Family', hue='Survived', data=df_train, ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize', y=1.02)

# (3)
df_train[['Family', 'Survived']].groupby(['Family'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending in Family', y = 1.02)                                                                                                                                 

plt.subplots_adjust(wspace=0.2, hspace = 0.5)
plt.show()

# (1)보면 가족 수는 1명 ~ 11명까지 있음. 대부분이 1명이고 2,3,4명이 많음
# (2),(3)을 보면 가족 수가 4명일 때 생존율이 높고, 너무 많거나 적은 경우 생존율이 낮음
fug, ax = plt.subplots(1, 1, figsize=(8,8))
g = sns.distplot(df_train['Fare'], color = 'b', label = 'Skewness : {:.2f}'. format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc = 'best')

# 그래프를 보면 데이터 분포가 비대칭적인 것을 알 수 있음 -> 이 상태로 모델에 입력하면 학습이 잘 안됨
# 이상치의 영향을 줄여주기 위해 fare칼럼 데이터에 모두 log를 취해줌 -> 판다스의 map이나 apply를 사용하여 lambda함수를 인자로 넣어줌
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean() # testset에 있는 null값을 평균값으로 치환

df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

fig, ax = plt.subplots(1, 1, figsize = (8,8))
g = sns.distplot(df_train['Fare'], color = 'b', label = 'Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)
g = g.legend(loc = 'best')

# 비대칭적인 부분이 완화됨
df_train.head()
df_train['Ticket'].value_counts()
# null값은 없지만 문자형이므로 전처리 필요
# 생존율과는 관계 없어보임
df_train['Name'].str  # 정규표현식 사용을 위해 Name 데이터 타입을 string으로 변환
df_train['Name'].str.extract('([A-Za-z]+)\.')
# A~Z,a~z까지 하나라도 있고, .가 있는 것 출력
df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.')
df_test['Initial']= df_test.Name.str.extract('([A-Za-z]+)\.')
df_train.head()
df_test.head()
pd.crosstab(df_train['Initial'],df_train['Sex']).T.style.background_gradient(cmap = 'summer_r')
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

# inplace=True로 원본 데이터에 바로 적용
df_train.groupby('Initial').mean() # 각 title별 평균나이 확인
df_train.loc[(df_train['Age'].isnull()) &(df_train['Initial']=='Mr'),'Age'] = 33
# Mr인 경우 평균나이가 32.79세이므로 33으로 채워줌
df_train.loc[(df_train['Age'].isnull()) &(df_train['Initial']=='Mrs'),'Age'] = 37
df_train.loc[(df_train['Age'].isnull()) &(df_train['Initial']=='Master'),'Age'] = 5
df_train.loc[(df_train['Age'].isnull()) &(df_train['Initial']=='Miss'),'Age'] = 22
df_train.loc[(df_train['Age'].isnull()) &(df_train['Initial']=='Other'),'Age'] = 45
df_test.loc[(df_test['Age'].isnull()) &(df_test['Initial']=='Mr'),'Age'] = 33
df_test.loc[(df_test['Age'].isnull()) &(df_test['Initial']=='Mrs'),'Age'] = 37
df_test.loc[(df_test['Age'].isnull()) &(df_test['Initial']=='Master'),'Age'] = 5
df_test.loc[(df_test['Age'].isnull()) &(df_test['Initial']=='Miss'),'Age'] = 22
df_test.loc[(df_test['Age'].isnull()) &(df_test['Initial']=='Other'),'Age'] = 45
df_train['Age'].isnull().sum()
df_test['Age'].isnull().sum()
df_train['Embarked'].isnull().sum()
df_train.shape
# 891 데이터 중 null 값은 2개 뿐이므로 가장 많은 데이터로 치환
df_train['Embarked'].fillna('S', inplace = True)
df_test['Embarked'].fillna('S', inplace = True)
df_train['Embarked'].isnull().sum()
df_test['Embarked'].isnull().sum()
df_train['Age_cat'] = 0
# 카테고리형으로 변경한 데이터를 삽입할 칼럼 추가
df_train.head()  # Age_cat 칼럼 추가됨
df_train.loc[df_train['Age']<10,'Age_cat'] = 0
df_train.loc[(10 <= df_train['Age']) & (df_train['Age'] < 20), 'Age_cat'] = 1
df_train.loc[(20 <= df_train['Age']) & (df_train['Age'] < 30), 'Age_cat'] = 2
df_train.loc[(30 <= df_train['Age']) & (df_train['Age'] < 40), 'Age_cat'] = 3
df_train.loc[(40 <= df_train['Age']) & (df_train['Age'] < 50), 'Age_cat'] = 4
df_train.loc[(50 <= df_train['Age']) & (df_train['Age'] < 60), 'Age_cat'] = 5
df_train.loc[(60 <= df_train['Age']) & (df_train['Age'] < 70), 'Age_cat'] = 6
df_train.loc[(70 <= df_train['Age']) , 'Age_cat'] = 7
df_test.loc[df_test['Age']<10,'Age_cat'] = 0
df_test.loc[(10 <= df_test['Age']) & (df_test['Age'] < 20), 'Age_cat'] = 1
df_test.loc[(20 <= df_test['Age']) & (df_test['Age'] < 30), 'Age_cat'] = 2
df_test.loc[(30 <= df_test['Age']) & (df_test['Age'] < 40), 'Age_cat'] = 3
df_test.loc[(40 <= df_test['Age']) & (df_test['Age'] < 50), 'Age_cat'] = 4
df_test.loc[(50 <= df_test['Age']) & (df_test['Age'] < 60), 'Age_cat'] = 5
df_test.loc[(60 <= df_test['Age']) & (df_test['Age'] < 70), 'Age_cat'] = 6
df_test.loc[(70 <= df_test['Age']) , 'Age_cat'] = 7
df_train.head()
df_test.head()
def category_age(x):
    if x < 10 :
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
df_test['Age_cat_2'] = df_test['Age'].apply(category_age)
(df_train['Age_cat'] == df_train['Age_cat_2']).all()
(df_test['Age_cat'] == df_test['Age_cat_2']).all()

# ()로 묶어주면 시리즈가 됨
# 전체가 같은지 확인할 때는 all사용

# all하면 모든 게 True일 때 True 나옴. 하나라도 True가 아니면 False
# any하면 하나라도 True가 있으면 True가 아님. True가 하나도 없으면 False
df_train.drop(['Age_cat_2'], axis =1, inplace = True)  # axis =1로 해주면 세로로 날아감df_test.drop(['Age_cat_2'], axis =1, inplace = True)
df_test.drop(['Age_cat_2'], axis =1, inplace = True) 
df_train.head()
df_test.head()
df_train.Initial.unique()
# mr는 1, mrs는 2 등으로 변경할 예정
df_train['Initial'] = df_train['Initial'].map({'Master' : 0, 'Miss' : 1, 'Mr' : 2, 'Mrs': 3,'Other':4})
df_test['Initial'] = df_test['Initial'].map({'Master' : 0, 'Miss' : 1, 'Mr' : 2, 'Mrs': 3,'Other':4})
df_train.head()
df_test.head()
df_train.Embarked.unique()
df_train['Embarked'].value_counts()
# unique한 것의 개수까지 알 수 있음
df_train['Embarked'] = df_train['Embarked'].map({'C':0, 'Q':1,'S':2})
df_test['Embarked'] = df_test['Embarked'].map({'C':0, 'Q':1,'S':2})
df_train.head()
df_test.head()
df_train['Sex'].unique()
df_train['Sex'] = df_train['Sex'].map({'female':0,'male':1})
df_test['Sex'] = df_test['Sex'].map({'female':0,'male':1})
df_train.head()
df_test.head()
heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'Family', 'Initial', 'Age_cat']] 

colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,
           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})

del heatmap_data

# astype(float): heatmap 데이터를 실수형으로 바꿔줌
# linewidths: 네모 칸 사이 라인의 크기
# vmax: 오른쪽 축
# linecolor: 네모 칸 사이 라인 색깔
# annot: 네모 칸 안에 숫자 표시
# annot_kws{'size':16}: 네모 칸 안에 숫자 크기
# fmt ='.2f': 네모 칸 안에 숫자 2번째 자리까지 반올림

# 우리목표는 Survived이니 x축의 survived 선택하고 y축 피처 마다의 계수를 보면 됨
# fare와 pclass의 계수가 1이라면 둘 중 하나의 피처만 선택해도 됨
df_train = pd.get_dummies(df_train,columns=['Initial'],prefix='Initial') # prefix=Initial: Initial 0, Initial 1 이렇게 구분하기 쉽게 나옴
df_train.head()
df_test = pd.get_dummies(df_test,columns=['Initial'],prefix='Initial')
df_test.head()
df_train = pd.get_dummies(df_train,columns=['Embarked'],prefix='Embarked')
df_train.head()
df_test = pd.get_dummies(df_test,columns=['Embarked'],prefix='Embarked')
df_test.head()
df_train.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Age'],axis = 1,inplace = True)
df_test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Age'],axis = 1,inplace = True)
df_train.head()
df_test.head()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn import metrics # 모델의 평가
X = df_train.drop('Survived', axis=1)  # survived를 제외한 피처들
Y = df_train['Survived']               # survived만 있는 피처
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=11)
dt = DecisionTreeClassifier(random_state=11)

# 학습
dt.fit(X_train , y_train)

# 예측
dt_pred = dt.predict(X_val)

# 평가
dt_accuracy = accuracy_score(y_val, dt_pred)
print('결정 트리 예측 정확도: {0:.4f}'.format(dt_accuracy))
# 하이퍼 파라미터 설정
parameters = {'max_depth':[2,3,5,10],'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]} 

# 하이퍼 파라미터를 5개의 train, val로 나누어 테스트 수행 설정
grid_dt = GridSearchCV(dt, param_grid = parameters, scoring = 'accuracy', cv=5, verbose=1 , refit = True)  #  verbose: 얼마나 자세히 정보를 표시할 것인가 0,1,2로 나눠짐

# 튜닝된 하이퍼 파라미터로 학습
grid_dt.fit(X_train, y_train)

# 최고 성능을 낸 하이퍼 파라미터 값과 그때의 평가값 저장
print('GridSearchCV 최적 하이퍼 파라미터:', grid_dt.best_params_)       # 최적 하이퍼 파라미터
print('GridSearchCV 최고 정확도:{0:.4f}'.format(grid_dt.best_score_))  # 최적 하이퍼 파라미터일 때 정확도
# refit = True로 최적 하이퍼 파라미터 미리 학습하여 best_estimator_로 저장됨(별도로 fit할 필요없음)
dt1= grid_dt.best_estimator_   

# 재예측
dt1_pred = dt1.predict(X_val)   

# 재평가
dt1_accuracy = accuracy_score(y_val , dt1_pred)
print('결정 트리 예측 정확도:{0:.4f}'.format(dt1_accuracy))
import seaborn as sns

ftr_importances_values = dt1.feature_importances_

# 정렬을 쉽게 하고, 막대그래프로 표현하기 위해 Series로 변환
ftr_importances = pd.Series(ftr_importances_values, index=X_train.columns  )

# top20만 확인
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

# 그래프 설정
plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20 , y = ftr_top20.index)

plt.show()
rf = RandomForestClassifier(random_state=11)

# 학습
rf.fit(X_train , y_train)

# 예측
rf_pred = rf.predict(X_val)

# 평가
rf_accuracy = accuracy_score(y_val ,rf_pred)
print('랜덤 포레스트 정확도: {0:.4f}'.format(rf_accuracy))
# 하이퍼 파라미터 설정
parameters = {'n_estimators':[10], 'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [8, 12, 18 ],'min_samples_split' : [8, 16, 20]}

# 하이퍼 파라미터를 2개의 train, val로 나누어 테스트 수행 설정
grid_rf = GridSearchCV(rf, param_grid = parameters , cv=2)     # 이번에는 refit 안해봄

# 튜닝된 하이퍼 파라미터로 학습
grid_rf.fit(X_train, y_train)

# 최고 성능을 낸 하이퍼 파라미터 값과 그때의 평가값 저장
print('최적 하이퍼 파라미터:\n', grid_rf.best_params_)           # 최적 하이퍼 파라미터
print('최고 예측 정확도: {0:.4f}'.format(grid_rf.best_score_))  # 최적 하이퍼 파라미터일 때 정확도
# 최적 하이퍼 파라미터 적용
rf1 = RandomForestClassifier(max_depth=8, min_samples_leaf=8, min_samples_split=8, random_state=0)

# 재학습
rf1.fit(X_train , y_train)    # refit 안했으므로 fit도 수행

# 재예측
rf1_pred = rf1.predict(X_val)

# 재평가
print(accuracy_score(y_val , rf1_pred))
import seaborn as sns

ftr_importances_values = rf1.feature_importances_

# Top 중요도로 정렬을 쉽게 하고, 막대그래프로 쉽게 표현하기 위해  sort_values사용해 Series로 변환
ftr_importances = pd.Series(ftr_importances_values,index=X_train.columns) 

# 중요도값 순으로 Series를 정렬
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]  # sort_values사용해 많은 피처들 중에 일부만 봄

# 그래프 설정
plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20 , y = ftr_top20.index)
plt.show()
# 개별 모델
lr = LogisticRegression()
rf = RandomForestClassifier(random_state=11)

# 개별 모델을 보팅 방식으로 결합
vo = VotingClassifier( estimators=[('LR',lr),('RF',rf)] , voting='soft' )

# 학습
vo.fit(X_train , y_train)

# 예측
vo_pred = vo.predict(X_val)

# 평가
accuracy = accuracy_score(y_val , vo_pred)
print('Voting 분류기 정확도: {0:.4f}'.format(accuracy))
gbm = GradientBoostingClassifier(random_state=0)

# 학습
gbm.fit(X_train , y_train)

# 예측
gbm_pred = gbm.predict(X_val)

# 평가
gbm_accuracy = accuracy_score(y_val, gbm_pred)

print('GBM 정확도: {0:.4f}'.format(gbm_accuracy))
# 하이퍼 파라미터 설정
parameters = {'n_estimators':[100, 500], 'learning_rate' : [ 0.05, 0.1]}   # 오래 걸리므로 많이 하지 않음

# 하이퍼 파라미터를 3개의 train, val로 나누어 테스트 수행 설정
grid_gbm = GridSearchCV(gbm , param_grid=parameters , cv=2 ,verbose=1, refit=True)     # cv도 2개만

# 튜닝된 하이퍼 파라미터로 학습
grid_gbm.fit(X_train , y_train)


# 최고 성능을 나타낸 하이퍼 파라미터의 값과 그때의 평가값 저장
print('최적 하이퍼 파라미터:\n', grid_gbm.best_params_)             # 최적 하이퍼 파라미터
print('최고 예측 정확도: {0:.4f}'.format(grid_gbm.best_score_))    # 최적 하이퍼 파라미터일 때 정확도
# refit = True로 최적 하이퍼 파라미터 미리 학습하여 best_estimator_로 저장됨(별도로 fit할 필요없음)
gbm1 = grid_gbm.best_estimator_

# 재예측
gbm1_pred = gbm1.predict(X_val)

# 재평가
gbm1_accuracy = accuracy_score(y_val, gbm1_pred)
print('GBM 정확도: {0:.4f}'.format(gbm1_accuracy))
import seaborn as sns

ftr_importances_values = gbm1.feature_importances_

# Top 중요도로 정렬을 쉽게 하고, 막대그래프로 쉽게 표현하기 위해  sort_values사용해 Series로 변환
ftr_importances = pd.Series(ftr_importances_values,index=X_train.columns) 

# 중요도값 순으로 Series를 정렬
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]  # sort_values사용해 많은 피처들 중에 일부만 봄

# 그래프 설정
plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20 , y = ftr_top20.index)
plt.show()
xgb = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)   

# 원래 evalution데이터는 검증 데이터로 해야하는데 테스트 데이터로 하는 이유는 현재 데이터 수가 적어서
evals = [(X_val,y_val)]  

# 학습 (조기 중단 적용)
xgb.fit(X_train, y_train, early_stopping_rounds=100, eval_metric= 'logloss', eval_set=evals, verbose=0)     # eval_metric는 성능 평가 방법, error/logloss 적용
                                                                                                               # eval_set: 성능 평가를 수행할 평가용 데이터 설정                     
# 예측
xgb_pred = xgb.predict(X_val)

# 평가
xgb_accuracy = accuracy_score(y_val,xgb_pred)   
print('accuracy: {0:.4f}'.format(xgb_accuracy))
# 하이퍼 파라미터 설정
parameters = {'max_depth':[5, 7] , 'min_child_weight':[1,3] }   

# 원래 evalution 데이터는 검증 데이터로 해야하는데 테스트 데이터로 하는 이유는 현재 데이터 수가 적어서
evals = [(X_val,y_val)]  

# 하이퍼 파라미터를 3개의 train, val로 나누어 테스트 수행 설정
grid_xgb = GridSearchCV(xgb, param_grid=parameters, cv=3)

# 튜닝된 하이퍼 파라미터로 학습
grid_xgb.fit(X_train, y_train, early_stopping_rounds=30, eval_metric='logloss', eval_set=evals, verbose=0)

# 튜닝된 하이퍼 파라미터로 예측
grid_xgb_pred = xgb.predict(X_val)

# 최고 성능을 나타낸 하이퍼 파라미터의 값과 그때의 평가값 저장
grid_xgb_accuracy = accuracy_score(y_val, grid_xgb_pred)   

print('최적 하이퍼 파라미터:\n', grid_xgb.best_params_)     # 최적 하이퍼 파라미터
print('accuracy: {0:.4f}'.format(grid_xgb_accuracy))     # 최적 하이퍼 파라미터일 때 정확도
# 최적 하이퍼 파라미터 적용
xgb1 = XGBClassifier(n_estimators=1000, random_state=156, learning_rate=0.02, max_depth=5, min_child_weight=1, colsample_bytree=0.75, reg_alpha=0.03)

# 원래 evalution 데이터는 검증 데이터로 해야하는데 테스트 데이터로 하는 이유는 현재 데이터 수가 적어서
evals = [(X_val,y_val)]  

# 재학습
xgb1.fit(X_train, y_train, early_stopping_rounds=200, eval_metric='logloss',eval_set=evals, verbose=0)

# 재예측                                                                                                                             
xgb1_pred = xgb1.predict(X_val)

# 재평가
xgb1_accuracy = accuracy_score(y_val, xgb1_pred)   
print('accuracy: {0:.4f}'.format(xgb1_accuracy))
from xgboost import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(1,1,figsize=(10,8))
plot_importance(xgb1, ax=ax , max_num_features=20,height=0.4)

# 칼럼이름이 나오는 이유는 numpy가 아닌 dataframe으로 했기 때문
lgbm = LGBMClassifier(n_estimators=400)

# evalution 세트는 검증 데이터 세트므로 테스트 데이터 사용하면 안됨(모의고사봐야하는데 본고사보면 안됨)
evals = [(X_val, y_val)]  

# 학습
lgbm.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", eval_set=evals, verbose=0)

# 예측
lgbm_pred = lgbm.predict(X_val)

# 평가
lgbm_accuracy = accuracy_score(y_val, lgbm_pred)
print('accuracy: {0:.4f}'.format(lgbm_accuracy))
# 하이퍼 파라미터 설정
parameters = {'num_leaves': [32, 64 ],'max_depth':[128, 160],'min_child_samples':[60, 100], 'subsample':[0.8, 1]}   # num_leaves: 하나의 트리가 가질수있는 최대 리프 개수
                                                                                                                # sumbsample: 과적합 방지로 데이터 샘플링하는 비율
                                                                                                                # min_child_samples: 리프 노드가 되기 위해 최소로 필요한 레코드 수, 과적합 방지
# evalution 세트는 검증 데이터 세트므로 테스트 데이터 사용하면 안됨(모의고사봐야하는데 본고사보면 안됨)
evals = [(X_val, y_val)] 

# 하이퍼 파라미터를 2개의 train, val로 나누어 테스트 수행 설정
grid_lgbm = GridSearchCV(lgbm, param_grid=parameters, cv=2)


# 하이퍼 파라미터로 학습과 평가
grid_lgbm.fit(X_train, y_train, early_stopping_rounds=30, eval_metric='logloss', eval_set=evals, verbose=0)

# 예측
grid_lgbm_pred = grid_lgbm.predict(X_val)

# 평가
grid_lgbm_accuracy = accuracy_score(y_val, grid_lgbm_pred)
print('accuracy: {0:.4f}'.format(grid_lgbm_accuracy))
lgbm1 = LGBMClassifier(n_estimators=1000, num_leaves=32, sumbsample=0.8, min_child_samples=100,max_depth=128)

evals = [(X_val, y_val)]

# 재학습
lgbm1.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=evals, verbose= 0)

# 재예측
lgbm1_pred = lgbm1.predict(X_val)

# 평가
lgbm1_accuracy = accuracy_score(y_val, lgbm1_pred)
print('accuracy: {0:.4f}'.format(lgbm1_accuracy))
from lightgbm import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(lgbm1, ax=ax)  # ax로 축도 가져옴, 학습이 완료된 lgbm_wrapper가져옴

# Column_1은 두번째 열 -> 데이터가 numpy라 칼럼명을 알 수 없어 순서로 알려줌
# 개별 ML 모델을 위한 Classifier 생성
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100, random_state=0)
gbm = GradientBoostingClassifier(random_state=0)
xgb = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)  
lgbm = LGBMClassifier(n_estimators=400)

# 최종 Stacking 모델을 위한 Classifier는 로지스틱 회귀 
lr_final = LogisticRegression(C=10)
# 개별 모델들을 학습
dt.fit(X_train, y_train)
rf.fit(X_train , y_train)
gbm.fit(X_train , y_train)
xgb.fit(X_train, y_train)
lgbm.fit(X_train, y_train)
# 학습된 개별 모델을 예측하고 각자 반환하는 예측 데이터 셋을 생성하고 개별 모델의 정확도 측정
dt_pred = dt.predict(X_val)
rf_pred = rf.predict(X_val)
gbm_pred = gbm.predict(X_val)
xgb_pred = xgb.predict(X_val)
lgbm_pred = lgbm.predict(X_val)

print('KNN 정확도: {0:.4f}'.format(accuracy_score(y_val, dt_pred)))
print('랜덤 포레스트 정확도: {0:.4f}'.format(accuracy_score(y_val, rf_pred)))
print('결정 트리 정확도: {0:.4f}'.format(accuracy_score(y_val, gbm_pred)))
print('에이다부스트 정확도: {0:.4f} :'.format(accuracy_score(y_val, xgb_pred)))
print('에이다부스트 정확도: {0:.4f} :'.format(accuracy_score(y_val, lgbm_pred)))
# 위에서 나온 5개 모델을 가로로 먼저 쌓음
pred = np.array([dt_pred, rf_pred, gbm_pred, xgb_pred, lgbm_pred])
print(pred.shape)

# 가로로 쌓은 모델을 세로로 쌓기 위해 transpose를 이용해 행과 열 위치 변경
# 컬럼 레벨로 각 알고리즘의 예측 결과를 피처로 만듦
pred = np.transpose(pred)
print(pred.shape)

# 가로로 만들어진 모델의 행과 위치 바꿈
# 학습을 테스트로 하여 과적합 됨, 위에서도 이렇게 함 -> 이를 해결하기 위해 CV사용함
lr_final.fit(pred, y_val) 
final = lr_final.predict(pred)

print('최종 메타 모델의 예측 정확도: {0:.4f}'.format(accuracy_score(y_val , final)))
lr = LogisticRegression()

# 학습
lr.fit(X_train , y_train)

# 예측
lr_pred = lr.predict(X_val)

# 평가
accuracy = accuracy_score(y_val, lr_pred)
print('accuracy: {:0.3f}'.format(accuracy_score(y_val, lr_pred)))
# 하이퍼 파라미터 설정
parameters={'penalty':['l2', 'l1'],'C':[0.01, 0.1, 1, 1, 5, 10]}  # c는 작을수록 규제가 큼

# 하이퍼 파라미터를 3개의 train, val로 나누어 테스트 수행 설정
grid_lr = GridSearchCV(lr, param_grid=parameters, scoring='accuracy', cv=3 , refit= True)

# 튜닝된 하이퍼 파라미터로 학습
grid_lr.fit(X_train , y_train)

# 최고 성능을 낸 하이퍼 파라미터 값과 그때의 평가값 저장
print('GridSearchCV 최적 하이퍼 파라미터:', grid_lr.best_params_)       # 최적 하이퍼파라미터
print('GridSearchCV 최고 정확도:{0:.4f}'.format(grid_lr.best_score_))  # 최적 하이퍼파라미터일 때 정확도
# refit = True로 최적 하이퍼 파라미터 미리 학습하여 best_estimator_로 저장됨(별도로 fit할 필요없음)
lr1 = grid_lr.best_estimator_   

# 재예측
lr1_pred = lr1.predict(X_val)         

# 재평가
lr1_accuracy = accuracy_score(y_val , lr1_pred)
print('결정 트리 예측 정확도:{0:.4f}'.format(lr1_accuracy))
prediction = xgb1.predict(X_val)  # 학습한 걸 바탕으로 val데이터로 예측해봄
prediction
submission = pd.read_csv('../input/titanic/gender_submission.csv')
prediction = xgb1.predict(df_test)  # 실제 예측
submission['Survived'] = prediction
submission['Survived'] = prediction  # survived에 내가 실제로 예측한걸 저장
submission.to_csv('submission.csv', index = False)  # 캐글 커널 서버에 csv파일 저장
