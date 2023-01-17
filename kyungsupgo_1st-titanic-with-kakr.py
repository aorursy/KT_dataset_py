import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





## matplotlib의 스타일 중 seaborn 스타일 사용

plt.style.use('seaborn')

# 일일히 폰트 사이즈 주는게 아니라 아래처럼하면 plot의 모든 #

#글자사이즈 2.5로 통일#

sns.set(font_scale=2.5)



# data set의 null데이터를 쉽1게 보여줄 수 있는 library

import missingno as msno

# 쓸모 없는 warning 무시하기 (warning 뜨는 거 전체 무시인듯)

import warnings

warnings.filterwarnings('ignore')



#inline으로 하면 새로운 창에 뜨는게 아니라 바로바로 보이게 하는거

%matplotlib inline

# input 은 하위디렉토리 말하고 

#보통 data set들은 다 input에 담아져 있음

df_train= pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head(10)
#shape은 ()없이 사용 이거 자주씀

df_train.shape

#위에 보면 행 891개인데 age의 행 count가 714

# 일단 age에 null있는 거 알 수 있음

# decribe 하고 head해서 내용 간단히 보는 거하고 둘다 해줘야 하는게 

# decribe는 연속형변수(숫자)에 대한 요약이라  이산형변수에 대한 내용은 아예 나타나질 않음

# 그러니 이산형변수 파악 까먹지 말고 head 도 반드시 진행할 것 

df_train.describe()
df_test.describe()
df_train.columns
for col in df_train.columns:

    #오른쪽 > 정렬 //  왼쪽 < 정렬 //없으면 정렬 안함

    

    # df_train[col].shape[0] :for에서 들어온거 col로 보내주면 df_train의 컬럼명의 shape->몇행 몇열인데 0이니까 행만

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col,100*(df_train[col].isnull().sum()/df_train[col].shape[0]))

    print(msg)
for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col,100*(df_test[col].isnull().sum()/df_test[col].shape[0]))

    print(msg)
# iloc -> index location(pandas문법)의 줄임말

# iloc : 원하는 위치에 있는 데이터프레임내의 것을 indexing하는 문법

# iloc[:]  :는 첨부터 끝까지 2: 는 2부터  [3:5] 3부터 4 마지막 제외

# 여기서 사실 iloc가 전체가 선택이니 그냥 df = df_train만 줘도 진행됨

# color는 RGB의 각각 순서대로 R,G,B 0~1까지 

msno.matrix(df= df_train.iloc[: , :],figsize=(8,8), color = (0.8,0.5,0.2))
#위에거는 어느 위치에 null인지 보기 편하고 아래거는 퍼센트를 좀 시각적으로 볼수있는 느낌

msno.bar(df= df_train.iloc[: , :],figsize=(8,8), color = (0.8,0.5,0.2))
# 1행2열짜리 밑바탕 그림(subplot)을 그려라 #figsize는 기본 바탕의 크기 저 사이즈를 좁히면 일정숫자 이하나 이상일경우 간격숫자도 바뀜

f,ax =plt.subplots(1,2,figsize=(18,8))



#df_train의 Survived칼럼의 각 value각의 count를 pie plot으로 그려라 

# explode ->사이간격벌리기 autopct=-> %의형

# ax에 2개가 들어있는데 ax=ax[0]은 왼쪽0 오른쪽 1의 순서대로 중에 어떠것을 pie차트로 할거냐 ax[0]번째거

#shadow는 그림자 

df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

#df_train['Survived'].value_counts()는 series형태이고 (type해봐) series형태들은 plot을 가지고 있다

#df_train['Survived'].value_counts().plot()으로 확인가능 == plt.plot(df_train('Survived').value_counts())



## ***번외로 위의것 pd.DataFrame(df_train['Survived'].value_counts()) 이렇게 하면 Series type이 아닌

## Data Frame type으로 가능 











#ax[0]의 title적고  label은 없이 가겠다

ax[0].set_title('Pie plot - Survied')

ax[0].set_ylabel('')



#seaborn(=sns)의 countplot는 컬럼이름(Survived)을 첫번째로 주고 data를 두번째 ax로 위치를 세번째

sns.countplot('Survived',data=df_train,ax=ax[1])

ax[1].set_title('Count plot - Survived')

plt.show()
# df_train[[]] 이렇게 리스트 형태로하고 안에 두개이상 쓰면pandas.core.frame.DataFrame 처럼 dataframe의 컬럼들이 나오고

# df_train[] 한개는 pandas.core.series.Series 처럼 series가 나옴

# 그래서 [] 한개짜리는 안에 여러개를 넣을 수가 없음 series니까 단일 한개에 대해서만 나옴

df_train[['Pclass','Survived']].groupby(['Pclass'], as_index= True).count()



#df_train[['Pclass','Survived']].groupby(['Pclass']) 까지만 쓰면 groupby를 pclass기준으로 하겟다.

# 그러면 data frame groupby 객체가 하나 만들어지는 것

# 만들어진 객체는 수많은 매소드를 가지고 있음 df_train[['Pclass','Survived']].groupby(['Pclass']).각종매소드



#부가

# crosstab 매소드 각각 cross table로 보여줌 

pd.crosstab(df_train['Pclass'],df_train['Survived'], margins=True)
#부가

# crosstab 에 스타일 주고 gradient 기울어지는 정도에 따라 색깔을 달리 하겠다. summer_r,jet,winter등 다양

# 종류는 color map scheme 를 google에 검색하면 많이 나옴

pd.crosstab(df_train['Pclass'],df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
# as_index하면 Pcalss를 인덱스 열로 둘거냐 말거냐 

df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).mean()

#만약 위처럼 한게 안그려지면

#df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False)#오름차순도 가능)
df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).mean().plot()

df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().plot()

# 위와 아래의 차이는 index로 여부에 따라 두개의 series가 있냐 없냐에 따라 그림이 변함 
df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=True).mean().plot.bar()

#보기 편하게 bar로도 가능 
#y_position 위치 지정때문에 미리 숫자대입해놓는 건데 사실상 안하고 직접입력해도 무방

y_position = 1.02

#앞전과 동일

f,ax= plt.subplots(1,2, figsize=(18,8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#DDFD00','#D3D3D3'], ax=ax[0])



# title 위치 정하려고 y= y_position 하는데 이거 안하고 그냥 고대로 숫자 대입해도 됨

ax[0].set_title('Number of passengers by Pcalss', y = y_position)



# hue없이면 Pclass의 빈도수만 but hue하면 Pclass에 따른 Survived의 빈도수   //

# 같은 Pclass내의 Survived(=hue)를 색깔별로 다르게 나타내겠다

sns.countplot('Pclass',data= df_train, ax=ax[1],hue='Survived')

# y_position 위와 동일

ax[1].set_title('Pclass: Survived cs Dead', y= y_position)

plt.show()
f,ax = plt.subplots(1,2, figsize= (18,8))

df_train[['Sex','Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue= 'Survived', data= df_train, ax= ax[1])

ax[1].set_title('Sex: Suvived vs Dead')

plt.show()
df_train[['Sex','Survived']].groupby(['Sex'],  as_index=False).mean()
pd.crosstab(df_train['Sex'],df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
# x축에 Pclass y축에 Survived   hue에 색깔넣을 기준 

sns.factorplot('Pclass','Survived',hue= 'Sex', data = df_train, size = 6, aspect = 1.5)



## 아래 표에서 보듯이 여성이 생존률이 높고 / Pclass가 1->2->3 순으로 생존률 안좋아짐

## 각  point에 세로선이 error bar 인데 

## error bar 크면 신뢰도가 낮다 // error bar가 작으면 신뢰도가 높다로 볼 수 있다

## error bar 오차구간 
# 클래스 별로 나눠져서 보이게 되는 것

# 아래에 col인자를 지우고 hue로 바꿔서 대체하면 한그래프로 다들어감

sns.factorplot(x='Sex', y= 'Survived' , col= 'Pclass', data = df_train,saturation =5,

             size=9, aspect = 1)
print('제일 나이 많은 탑승객 {:.1f} years'.format(df_train['Age'].max()))

print('제일 어린 탑승객 : {:.1f}years'.format(df_train['Age'].min()))

print('탑승객 평균 나이 : {:.1f}years'.format(df_train['Age'].mean()))
fig, ax = plt.subplots(1, 1, figsize=(9, 5))



# kdeplot --> seaborn 스타일중 하나로 1차원 실수 분포 plot임(커널밀도 추정함수)

# https://blog.naver.com/loiu870422/220660847923 이곳에 잘나옴 

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

# df_train['Survived'] == 1 해당만족하는 조건만 TRUE값으로 출력되는데

# df_train[  ---- ] []안에 넣어주면 TRUE인 값만 반환해줌  **결국 조건으로 filtering하는 거



plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
#마찬가지로 kdeplot그릴건데 다른방법으로 그릴거임

plt.figure(figsize=(8,6))

# Pclass가 1인사람들의 age를 커널밀도 함수(kde) 로 그린것

df_train['Age'][df_train['Pclass']==1].plot(kind='kde')

df_train['Age'][df_train['Pclass']==2].plot(kind='kde')

df_train['Age'][df_train['Pclass']==3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])
#마찬가지로 histplot그릴건데 다른방법으로 그릴거임

plt.figure(figsize=(8,6))

# Pclass가 1인사람들의 age를 히스토그램 함수(hist) 로 그린것

#겹치면 안보임 

df_train['Age'][df_train['Pclass']==1].plot(kind='hist')

df_train['Age'][df_train['Pclass']==2].plot(kind='hist')

df_train['Age'][df_train['Pclass']==3].plot(kind='hist')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])
fig, ax = plt.subplots(1, 1, figsize=(9, 5))





# Pclass가 1일 때 죽은 사람 , 산 사람 각각 본거 

sns.kdeplot(df_train[  (df_train['Survived'] == 1) & (df_train['Pclass']==1) ]['Age'], ax=ax)

sns.kdeplot(df_train[  (df_train['Survived'] == 0) & (df_train['Pclass']==1) ]['Age'], ax=ax)



plt.legend(['Survived == 0', 'Survived == 1'])

plt.title('1st Pclass')

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(9, 5))





# Pclass가 2일 때 죽은 사람 , 산 사람 각각 본거 

sns.kdeplot(df_train[  (df_train['Survived'] == 1) & (df_train['Pclass']==2) ]['Age'], ax=ax)

sns.kdeplot(df_train[  (df_train['Survived'] == 0) & (df_train['Pclass']==2) ]['Age'], ax=ax)



plt.legend(['Survived == 0', 'Survived == 1'])

plt.title('2nd Pclass')

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(9, 5))





# Pclass가 3일 때 죽은 사람 , 산 사람 각각 본거 

sns.kdeplot(df_train[  (df_train['Survived'] == 1) & (df_train['Pclass']==3) ]['Age'], ax=ax)

sns.kdeplot(df_train[  (df_train['Survived'] == 0) & (df_train['Pclass']==3) ]['Age'], ax=ax)



plt.legend(['Survived == 0', 'Survived == 1'])

plt.title('3rd Pclass')

plt.show()
#age의 range를 다르게 할때 어떻게 나올까 보려고 하는 것

change_age_range_survival_ratio = []



# 1~80살까지 생존률 

for i in range(1,80):

    change_age_range_survival_ratio.append( df_train[ df_train['Age'] < i ]['Survived'].sum() / len(df_train[df_train['Age']<i]['Survived']) )



plt.figure(figsize=(7,7))

plt.plot(change_age_range_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

    


f,ax = plt.subplots(1,2, figsize= (18,8))



# split을 False로 하면 붙어있는게 떨어져서 그려짐  - 이게 바이올린 모양이라 이름

# sclae 은 area,count,width 이렇게 3종류 인자받을 수 있음

# area로 하면 같은넓이 내에서 하게되서 distribution을 각 해당개체별로 알수있음

# count

sns.violinplot('Pclass','Age',hue='Survived', data= df_train, scale='width', split=True, ax=ax[0])



ax[0].set_title('Pclass and Age vs Survived')

#y축 0부터 110 까지 10단위고 구간 

ax[0].set_yticks(range(0,110,10))





sns.violinplot('Sex','Age',hue='Survived', data= df_train, scale='count', split=True, ax=ax[1])



ax[1].set_title('Pclass and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
f, ax= plt.subplots(1,1, figsize=(7,7))

# sort_values 앞의 평균을 구하것을 by기준으로 정렬

df_train[['Embarked','Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar(ax=ax)
f,ax= plt.subplots(2,2, figsize=(20,15))



# 위가 2,2 의 도화지니까 ax도 0번재 로우 0번째 행렬에 그린다의 의미

sns.countplot('Embarked', data=df_train, ax=ax[0,0])

ax[0,0].set_title('(1) No. Of Passengers Boarded')



sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax[0,1])

ax[0,1].set_title('(2) Male-Female Split for Embarked')



sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1,0])

ax[1,0].set_title('(3) Embarked vs Survived')



sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1,1])

ax[1,1].set_title('(4) Embarked vs Pclass')



# ()안의 인자들은 좌우 , 상하 간격맞춰주는 것 

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()

# Pandas Series 형태끼리는 더하기 가능 (물론 같은 int형태 )

# 본인도 포함 위해 +1 해주기

df_train['FamilySize']=df_train['SibSp']+ df_train['Parch']+1

df_test['FamilySize']= df_test['SibSp']+ df_test['Parch']+1
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

# displot은 series그리면 series의 histogram그려주는 것

# Skewness 왜도  좌측-> 양수 //우측 -> 음수

g = sns.distplot(df_train['Fare'], color='b', label='Skewness: {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')





# 현재 왜도가 한쪽으로 치우쳐진 상태라 이대로 모델에 대입하면 모델이 잘못학습하여 성능이 낮아질 수 있다

# 모델이라는게 dataset을 표현하는 건데 치우쳐진 상태에서 쓰면 안된다. 

# 여러가지 작업으로 skewness를 조정해주는데 

# 여기서는 log를 취해서 진행 

#log취해주기 

# lamda - 한줄짜리 funtion method

# 특정 series의 value들의 동일한 opertation적용하려면 map과 apply를 쓰고 함수를 넣어주면 됨
# test set에 null값있는 것 평균값으로 치환  

df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()

# i가 0보다 크면 np.log(i) 적용 아니면 그냥 0

df_train['Fare']= df_train['Fare'].map(lambda i: np.log(i) if i>0 else 0)



# test set도 같이 진행

df_test['Fare']= df_test['Fare'].map(lambda i: np.log(i) if i>0 else 0)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
#Cabin은 null data가 너무 많아 여기서는 빼고 진행
df_train['Ticket'].value_counts()
df_train['Age'].isnull().sum()
df_train['Age'].count()
df_train['Name'].head()
#extract 정규표현식 함수 ->대/소문자 A부터 z까지 1개이상의 단어에 .이 붙은거 ex) Mr. Mrs. Ms.

df_train['initial']=df_train['Name'].str.extract('([A-Za-z]+)\.')

df_test['initial']=df_test['Name'].str.extract('([A-Za-z]+)\.')
pd.crosstab(df_train['initial'],df_train['Sex']).T.style.background_gradient(cmap='summer_r')

#해당 결과를 가지고 몇가지 결과로 압축 
#initial에 있는 것들 [] 앞에 있는 것들을 []뒤에 내용으로 치환하겠다.

# 1:1치환

# 뒤에 inplace인자가  df_train['initial']=df_train['initial'].replace이렇게 =을 따로 해줄필요없이

# 한번에 바꿔주는게 inplace인자

df_train['initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)



df_test['initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
df_train.groupby(['initial']).mean()
# 강의에서는 df_train.groupby('initial')['Survived'].mean().plot.bar() 이렇게 적었는데

#아래는 내가 적은거 둘이 나타내는 건 똑같음

df_train.groupby('initial').mean()['Survived'].plot.bar()
# 본격적으로 null value찾을 거고 train과 test 두개 data set사용해서 찾을거고

# 데이터 프레임 합치는 건 concat 과 merge

# 둘이 비슷한 기능인데 concat을 데이터프레임을 쌓는 느낌 merge는 같은 컬럼에 추가하는 방식

df_all=pd.concat([df_train,df_test])
# 그냥 concat하면 index가 순서대로 합쳐지는게 아니라 단순 병합만되어서

# reset_index로 index number 조정

# drop인자는 별도의 index열을 만드는게 아니라 

# index자체 번호를 바꾸고 싶을때 True로 주는 인자

df_all.reset_index(drop=True)

df_all.groupby(['initial']).mean()
# loc 원하는 부분 indexing하는 함수

# loc안에 조건들 들어간거고 ,하고 Age는 앞의 조건들 만족하는 것들중에 Age컬럼만 보여줌

# 따라서 해당조건을 만족하는 Age컬럼에 Mr인사람들의 평균치인 33세가 들어감

df_train.loc[(df_train['Age'].isnull()) & (df_train['initial'] == 'Mr') , 'Age'] =33
df_train.loc[(df_train.Age.isnull())&(df_train.initial=='Mr'),'Age'] = 33

df_train.loc[(df_train.Age.isnull())&(df_train.initial=='Mrs'),'Age'] = 37

df_train.loc[(df_train.Age.isnull())&(df_train.initial=='Master'),'Age'] = 5

df_train.loc[(df_train.Age.isnull())&(df_train.initial=='Miss'),'Age'] = 22

df_train.loc[(df_train.Age.isnull())&(df_train.initial=='Other'),'Age'] = 45



df_test.loc[(df_test.Age.isnull())&(df_test.initial=='Mr'),'Age'] = 33

df_test.loc[(df_test.Age.isnull())&(df_test.initial=='Mrs'),'Age'] = 37

df_test.loc[(df_test.Age.isnull())&(df_test.initial=='Master'),'Age'] = 5

df_test.loc[(df_test.Age.isnull())&(df_test.initial=='Miss'),'Age'] = 22

df_test.loc[(df_test.Age.isnull())&(df_test.initial=='Other'),'Age'] = 45
#Age의 null data가 정상적으로 모두 들어간것을 확인했다.

df_train['Age'].isnull().sum()
df_train['Embarked'].isnull().sum()

# 2개 밖에 안되니 그냥 대체
# fillna ->  null값 채워주는 함수

df_train['Embarked'].fillna('S', inplace=True)
# continuous feature를 categorical feature로 만들려고한다.

# 위의 과정을 자칫 잘못하면 정보손실 발생가능

#어떤생황에서는 오히려 잘 맞을 수도 있음

df_train['Age_cat'] = 0
df_train.head()
#나이를 10단위의 구간으로 자르기 



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
### 이 위에까지가 hard cording한 방법 

### 아래부터는 같은내용인데 함수 사용할거 
# 함수생성

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
# all 모든게 True일때 Ture반환

# any는 하나라도 True가 있으면 True 반환

(df_train['Age_cat'] == df_train['Age_cat_2']).all()
#이제 다 해봤으니 필요한 Age_cat만 남기고 날리기

# axis =1 -> 축을 세로로 둬서 세로줄 자체가 날라가는거 

df_train.drop(['Age','Age_cat_2'], axis=1 , inplace= True )

df_test.drop(['Age'], axis=1 , inplace= True)
df_train['initial'].unique()

# 해당 value들을 변환
# mapping 시켜줌 (치환)

df_train['initial'] = df_train['initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_test['initial'] = df_test['initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
# 아래줄과 해당줄은 카테고리 뭐있는지 볼때 쓰는 것들

# 위애건 numpy array 아래건 panda series type

df_train['Embarked'].unique()
df_train['Embarked'].value_counts()
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df_train['Sex'].unique()
df_train['Sex']=df_train['Sex'].map({'female':0,'male':1})

df_test['Sex']=df_test['Sex'].map({'female':0,'male':1})
# 상관관계 보려고 pearson correaltion구하는 과정

# 필요한 컬럼 가져와서 heatmap그리고

heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'initial', 'Age_cat']] 



# matplt

#colormap 은 색깔의 몸통 ? 이런느낌이래 

colormap = plt.cm.viridis

plt.figure(figsize=(10, 8))

plt.title('Pearson Correlation of Features', y=1.05, size=15)



# astype-> heatmap내의 자료들을 모두 float형태도 바꿔주겠다

# corr -> pandas dataframe의 correlation을 모두 구해줌

# linewidths -> 네모 칸마다의 거리

# vmax ->샐깔 범위의 변화

# annot ->안의 숫자

# fmt -> 안의 숫자 자리수 



sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,

           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16},fmt='.2f')



del heatmap_data
# 위에 그림에서 유추할 수 있는것 

# 그림에서 보았을때 생각보다 강한 상관관계를 그리고 있는 것이 없다.

# 이것은 우리가 모델을 학습시킬 때, 불필요한(redundant, superfluous) feature 가 없다는 것을 의미

# 1 또는 -1 의 상관관계를 가진 feature A, B 가 있다면, 우리가 얻을 수 있는 정보는 사실 하나

# redundant, superfluius에 대해서는 조금 더 공부하기 
df_train.head()
df_test.head()
# pandas의 one-hot encording 함수 get_dummies사용

#dataframe을 직접다룰때는 이거쓰는게 낫고 

# sklearn.preprocessing에도 Onehotencording 이 따로 있음

#prefix -->

# get_dummies쓰면 initial column이 날아간다 

df_train = pd.get_dummies(df_train, columns=['initial'], prefix='initial')

df_test = pd.get_dummies(df_test, columns=['initial'], prefix='initial')



# 카테고리가 많아서 one-hot encording시  column이 100개처럼 과도하게 많아지는 경우

# 오히려 과도한 columns때문에 학습에 지장을 줄 수 있음. 그래서 이런경우 다른방식의 encording 방법있음

# 유한님의 다른 방송에 있음 포르투 competition 할때 겪었던 문제래
#이제 머신러닝 모델 세워서 작업들어갈건데 그 전에 데이터 정리가 필요함 

#이제부터 정리시작 하고 모델 학습시작 

df_train.head()

# passengerID 미사용으로 제거

# Names, Ticket,Cabin미사용으로 제거

# Sibsp,Parch 는 family columns 만들었으므로 제거
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
#axis를 1로 해야 모두 날아간대

df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_train.head()
# 이진분류 문제라서 classfier # 

from sklearn.ensemble import RandomForestClassifier

# metric 모델평가하는 함수들 주는 것 

from sklearn import metrics

# train set , valid set test set나눌때

# train set을 train 과 valid 두개로 나눈다 

from sklearn.model_selection import train_test_split
# 타겟이 Survived니까 제외하고 열기준이라 axis가 1

X_train= df_train.drop(['Survived'],axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values



#위에처럼하면 dataframe이 아닌 arry배열로 들어감
# y_tr이 target label 인데 이게 지도 학습이라 

# X_tr으로 유추한 Survived의 값 즉, 타겟값을 y_tr에 넣어주는것

# train data 891개중  는 vld를  30 퍼센트만 사용 나머지를 X_tr에 사용 

X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)
# RandomForest의 Classifier사용

model = RandomForestClassifier()

#X_tr과 y_tr이용 학습

model.fit(X_tr, y_tr)

# vld set으로 예측

prediction = model.predict(X_vld)
# 윗줄의 prediction = model.predict(X_vld) 실제예측값하고 우리가 따로 뗴어놓은 y_vld랑 비교에서 

# 얼마나 정확한지 확인하는 코드

print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))

from pandas import Series

#model.feature_importances_ 이거는 학습시킨 모델은 무조건 importance를 가짐 

feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')

plt.show()
submission = pd.read_csv('../input/gender_submission.csv')