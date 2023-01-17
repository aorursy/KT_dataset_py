import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')

sns.set(font_scale=2.5) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.

import missingno as msno



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
# 라이브러리 import 해주는 과정.

# 시각화 도구(matplotlib,seaborn, plotly)

# 데이터 분석 도구(pandas, numpy)
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# [1] 데이터셋 확인

# null data가 있는지 확인.

# train, testset의 데이터셋을 pandas를 사용해 분석.
df_train.head()
# 여기에서 다룰 feature는 Pclass(승객의 클래스), Age(성별), SibSp(함께 탑승한 형제, 배후자)

# Parch(함께 탑승한 부모, 아이의 수), Fare(탑승료) 이며

# 예측하려는 target label은 Survived(생존 여부) 입니다.
df_train.describe()
# describe() 메소드를 이용해 각 feature 가 가진 통계치들을 반환.
df_test.describe()
# 여기에서 보다시피, null data가 존재하는 열(feature)이 존재.

# 이를 좀 더 보기 편하도록 그래프로 시각화.
for col in df_train.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)
for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))

    print(msg)
# Train, Test set 에서 Age에 20%, Cabin에 80%, train의 Embarked에 0.22% null data 존재하는 것을 볼 수 있습니다.



# missingno(msno) 라는 라이브러리를 사용하면 null data의 존재를 더 쉽게 볼 수 있습니다.
msno.matrix(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
msno.bar(df=df_train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
msno.bar(df=df_test.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
# Target label 확인

# target label 이 어떤 분포를 가지고 있는 지 확인 
f, ax = plt.subplots(1, 2, figsize=(18, 8))



df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived')



plt.show()
# 38.4 % 가 살아남았습니다.

# target label 의 분포가 제법 균일(balanced)합니다. 

# 불균일한 경우, 예를 들어서 100중 1이 99, 0이 1개인 경우, 모델이 모든것을 1이라 해도 정확도가 99%가 나오게 되고, 이 모델은 원하는 결과를 줄 수 없다.
# (2.) Exploratory data analysis(EDA)

# 방대한 데이터를 시각화 라이브러리 matplotlib, seaborn, plotly 를 이용해

# 본격적으로 데이터 분석을 해보겠습니다.
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
# Pclass(승객의 클래스) 에 따른 생존률의 차이. 

#'Pclass', 'Survived' 를 가져온 후, pclass 로 묶습니다.

# count() 를 하면, 각 class 에 몇명이 있는 지 확인할 수 있으며, 
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
# sum() 을 하면,그 중 생존한(survived=1)사람의 총합.
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
# 각 pclass 마다 0, 1 이 count가 되는데, 이를 평균내면 각 pclass 별 생존률이 나옵니다
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
# 보다시피, Pclass 가 좋을 수록(1st > 2nd > 3rd) 생존률이 높은 것을 확인할 수 있습니다.

# seaborn 의 countplot 을 이용해 특정 label 에 따라 그래프로 확인해볼 수 있습니다.
y_position = 1.02

f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])

ax[0].set_title('Number of Passengers By Pclass', y=y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()
# 클래스가 높을 수록, 생존 확률이 높은걸 확인할 수 있습니다. 

# Pclass 1, 2, 3 순서대로 63%, 48%, 25% 입니다

# 생존에 Pclass가 큰 영향을 미친다고 생각해볼 수 있으며, 이후 이 feature 를 사용하는 것이 좋을 것이라 판단
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
# 성별로 생존률이 어떻게 달라지는 지 확인.

# 보시다시피, 여자가 생존할 확률이 높습니다.
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
# Pclass 와 마찬가지로, Sex 도 예측 모델에 쓰일 중요한 feature 임을 알 수 있습니다
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, 

               size=6, aspect=1.5)
# Sex, Pclass 두가지에 관하여 생존이 어떻게 달라지는 지 확인.

# factorplot 을 이용하면, 손쉽게 3개의 차원으로 이루어진 그래프를 그릴 수 있습니다.



# 또한 성별 상관없이 클래스가 높을 수록 살 확률이 높고, 모든 클래스에서 여성의 생존률이 높음을 확인.
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))

print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))

print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))
# Age(나이)에 따른 특징
# 생존에 따른 Age의 histogram 을 그려보겠습니다.

# 생존자 중 10~30대의 경우가 많음을 볼 수 있습니다.
# Age distribution withing classes

plt.figure(figsize=(8, 6))

df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])
# Class 가 높을 수록 나이 많은 사람의 비중이 커짐.



# 나이범위를 점점 넓혀가며, 생존률이 어떻게 되는지 한번 봅시다.
cummulate_survival_ratio = []

for i in range(1, 80):

    cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))

    

plt.figure(figsize=(7, 7))

plt.plot(cummulate_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
# 나이가 어릴 수록 생존률이 확실히 높은것을 확인할 수 있습니다.

# 따라서 나이가 중요한 feature 로 쓰일 수 있음을 확인했습니다.



# 지금까지 본, Sex, Pclass, Age, Survived 모두에 대해서 보고싶습니다.

# x 축은 우리가 나눠서 보고싶어하는 case(여기선 Pclass, Sex) 를 나타내고,

# y 축은 보고 싶어하는 분류 (Age) 입니다.
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=df_train, scale='count', split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=df_train, scale='count', split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
# 그림은 클래스와 성별 각각의 경우에서 나이와 생존여부를 구분한 그래프입니다.

# 생존만 봤을 때, 모든 클래스에서 나이가 어릴 수록 생존.

# 오른쪽 그림에서 보면, 명확하게 여자가 생존을 많이 한것을 볼 수 있습니다.

# 따라서 여성과 아이를 먼저 챙긴 것을 분석 할 수 있습니다.
# Embarked 는 탑승한 항구를 나타냅니다.

# 위에서 해왔던 것과 비슷하게 탑승한 곳에 따라 생존률을 보겠습니다.
f, ax = plt.subplots(1, 1, figsize=(7, 7))

df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
# 보시다시피, C가 제일 높은 정도의 조금의 차이는 있지만 생존률은 좀 비슷한 거 같습니다.

# 다른 케이스와 연계해서 살펴보겠습니다.
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
# Figure(1) - 항구 별 탑승객의 수. 전체적으로 봤을 때, S 에서 가장 많은 사람이 탑승했습니다.

# Figure(2) - 항구 별 탑승객의 성별 비율. C와 Q 는 남녀의 비율이 비슷하고, S는 남자가 더 많습니다.

# Figure(3) - 항구 별 생존 확률. 생존확률이 S 경우 많이 낮은 걸 볼 수 있습니다.

# Figure(4) - 항구 별 클래스의 비율. C가 생존확률이 높은건 클래스가 높은 사람이 많이 타서 그렇습니다.

# 따라서 S는 3rd class 가 많아서 생존확률이 낮게 나옴을 알 수 있습니다.
# SibSp(형제, 자매) 와 Parch(부모, 자녀)를 합하면 Family 가 될 것입니다.

# Family 로 합쳐서 분석해봅시다.
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
print("Maximum size of Family: ", df_train['FamilySize'].max())

print("Minimum size of Family: ", df_train['FamilySize'].min())
# FamilySize 와 생존의 관계를 한번 살펴봅시다
f,ax=plt.subplots(1, 3, figsize=(40,10))

sns.countplot('FamilySize', data=df_train, ax=ax[0])

ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)



sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
# Figure (1) - 가족크기가 1~11까지 있음을 볼 수 있습니다. 대부분 1명이고 그 다음으로 2, 3, 4명입니다.

# Figure (2), (3) - 가족 크기에 따른 생존비교입니다. 

# 가족이 4명인 경우가 가장 생존확률이 높습니다. 가족수가 많아질수록, (5, 6, 7, 8, 11)

# 생존확률이 낮아진다. 가족수가 너무 작아도(1), 너무 커도(5, 6, 8, 11) 생존 확률이 작다.

# 3~4명 선에서 생존확률이 높은 걸 확인할 수 있습니다.
# Fare 는 탑승요금이며, contious feature 입니다. 한번 histogram 을 그려보겠습니다.
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
# 보시다시피, 분포가 매우 비대칭인 것을 알 수 있습니다.(high skewness).

# 만약 이대로 모델에 넣어준다면 자칫 모델이 잘못 학습할 수도 있습니다.

# outlier의 영향을 줄이기 위해 pandas의 유용한 기능을 이용해 Fare 에 log 를 취하겠습니다.
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean() # testset 에 있는 nan value 를 평균값으로 치환합니다.



df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
# log 를 취하니, 이제 비대칭성이 많이 사라진 것을 볼 수 있습니다.
# [3] Freature engineering



# 본격적인 feature engineering의 시작.

# 가장 먼저, dataset 에 존재하는 null data를 채운다.

# 아무 숫자로 채울 수는 없고, null data 를 포함하는 feature 의 statistics 를 참고하거나,

# 다른 아이디어를 짜내어 채울 수 있습니다.

# null data 를 어떻게 채우느냐에 따라 모델의 성능이 좌지우지될 수 있기 때문에, 신경써줘야할 부분입니다.
import numpy as np

import pandas as pd

from pandas import Series

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')

sns.set(font_scale=2.5) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



#ignore warnings

import warnings

warnings.filterwarnings('ignore')







%matplotlib inline



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다



df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()



df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
# Age 에는 null data가 177개나 있습니다. 

# 영어에서는 Miss, Mrr, Mrs 같은 title이 존재합니다. 이를 사용.



# pandas series 에는 data 를 string 으로 바꿔주는 str method, 정규표현식을 적용하게 해주는 extract method가 있습니다. 이를 사용하여 title 을 쉽게 추출할 수 있습니다.
df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

    

df_test['Initial']= df_test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
# pandas 의 crosstab 을 이용하여 우리가 추출한 Initial 과 성별 간의 count 를 살펴봅시다.
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex
# table 을 참고하여, 남자, 여자가 쓰는 initial 을 구분해 보겠습니다.

# replace 메소드를 사용하면, 특정 데이터 값을 원하는 값으로 치환해줍니다.
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)



df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
df_train.groupby('Initial').mean()
# 여성과 관계있는 Miss, Mrs 가 생존률이 높은 것을 볼 수 있습니다.
df_train.groupby('Initial')['Survived'].mean().plot.bar()
# 이제 본격적으로 Null 을 채울 것입니다. null data 를 채우는 방법은 정말 많이 존재합니다.

# train data의 값을 이용해 Age의 평균을 Null value 에 채우도록 하겠습니다.
df_train.groupby('Initial').mean()
# 아래 코드 첫줄을 해석하자면, isnull() 이면서 Initial 이 Mr 인 조건을 만족하는

# row(탑승객) 의 'Age' 의 값을 33으로 치환한다 입니다.
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
# Embarked 는 Null value 가 2개이고, S 에서 가장 많은 탑승객이 있었으므로,

# 간단하게 Null 을 S로 채우겠습니다.
df_train['Embarked'].fillna('S', inplace=True)
# 나이는 연속된 값(continuous feature)입니다. 여기에서는 이를 몇개의 그룹으로 나누어 카테고리화 시키겠습니다. 
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
# 간단한 함수를 만들어 apply 메소드에 넣어주는 방법입니다.
print(' ', (df_train['Age_cat'] == df_train['Age_cat_2']).all())
df_train.drop(['Age', 'Age_cat_2'], axis=1, inplace=True)

df_test.drop(['Age'], axis=1, inplace=True)
#현재 이름 이니셜은 Mr, Mrs, Miss, Master, Other 총 5개로 이루어져 있습니다. 

#카테고리로 이루어진 데이터를 먼저 컴퓨터가 인식할 수 있도록 수치화 시켜야 합니다.
df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
# Embarked 도 C, Q, S로 이루어져 있습니다. map 을 이용해 바꿔봅시다.
df_train['Embarked'].unique()
df_train['Embarked'].value_counts()
# 위 두 방법을 사용해 Embarked가 S, C, Q 세가지로 이루어진 것을 볼 수 있습니다. 이제 map을 사용해봅시다
df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
# 한번 Null 이 사라졌는지 확인해봅시다. 

#isnull() 메소드를 사용해 Series의 값들이 null 인지 아닌지에 대한 boolean 값을 얻을 수 있습니다. 

# 그리고 이것에 any() 를 사용하여, True 가 단하나라도 있을 시(Null이 한개라도 있을 시)

# True 를 반환해주게 됩니다. 우리는 Null 을 S로 다 바꿔주었으므로 False 를 얻게 됩니다
df_train['Embarked'].isnull().any()
# Sex 도 Female, male 로 이루어져 있습니다. map 을 이용해 바꿔봅시다.
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})

df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
# 이제 각 feature 간의 상관관계를 한번 보려고 합니다. 두 변수간의 Pearson correlation 을 구하면

# (-1, 1) 사이의 값을 얻을 수 있습니다. -1로 갈수록 음의 상관관계, 1로 갈수록 양의 상관관계를 의미하며,

# 0은 상관관계가 없다는 것을 의미합니다. 
heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']] 



colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,

           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})



del heatmap_data
# 우리가 EDA에서 살펴봤듯이, Sex 와 Pclass 가 Survived 에 상관관계가 어느 정도 있음을 볼 수 있습니다.

# 생각보다 fare 와 Embarked 도 상관관계가 있음을 볼 수 있습니다.

# 또한 우리가 여기서 얻을 수 있는 정보는 서로 강한 상관관계를 가지는 feature들이 없다는 것입니다.



#이제 실제로 모델을 학습시키기 앞서서 data preprocessing (전처리)을 진행해보겠습니다.
df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')
# 수치화시킨 카테고리 데이터를 그대로 넣어도 되지만, 모델의 성능을 높이기 위해

# one-hot encoding을 해줄 수 있습니다.

# 수치화는 간단히 Master == 0, Miss == 1, Mr == 2, Mrs == 3, Other == 4 로 매핑해주는 것을 말합니다.
df_train.head()
# 보시다시피 오른쪽에 우리가 만들려고 했던 one-hot encoded columns 가 생성된 것이 보입니다.



# Embarked 에도 적용하겠습니다.
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
# 필요한 columns 만 남기고 다 지웁시다.
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_train.head()
df_test.head()
# 보시다시피, train 의 Survived feature(target class)를 빼면 train, test 둘다 같은 columns 를 가진 걸 확인할 수 있습니다.
# 이제 준비가 다 되었으니 sklearn 을 사용해 본격적으로 머신러닝 모델을 만들어 봅시다.
#importing all the required ML packages

from sklearn.ensemble import RandomForestClassifier # 유명한 randomforestclassfier 입니다. 

from sklearn import metrics # 모델의 평가를 위해서 씁니다

from sklearn.model_selection import train_test_split # traning set을 쉽게 나눠주는 함수입니다.
# 가장 먼저, 학습에 쓰일 데이터와, target label(Survived)를 분리합니다. drop 을 사용해 간단히 할 수 있습니다.
X_train = df_train.drop('Survived', axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values
# 보통 train, test 만 언급되지만, 실제 좋은 모델을 만들기 위해서 우리는 valid set을 따로 만들어 모델 평가를 해봅니다.
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)
# 여기서 랜덤포레스트 모델을 사용하는데, 랜덤포레스트는 결정트리기반 모델이며 기본 default 세팅으로 진행하겠습니다.
model = RandomForestClassifier()

model.fit(X_tr, y_tr)

prediction = model.predict(X_vld)
print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
# 기본 모델을 활용하여, 82% 정도의 정확도를 보여줌.
# 학습된 모델은 feature importance 를 가지게 되는데, 우리는 이것을 확인하여 지금 만든 모델이 어떤 feature 에 영향을 많이 받았는 지 확인할 수 있습니다.
from pandas import Series



feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')

plt.show()
# 우리가 얻은 모델에서는 Fare 가 가장 큰 영향력을 가지며, 그 뒤로 Sex, Age_cat, FamilySize 가 차례로 중요도를 가집니다

# feature importance 는 지금 모델에서의 importance 를 나타냅니다. 만약 다른 모델을 사용하게 된다면 feature importance 가 다르게 나올 수 있습니다.

# 실제로 Fare 가 중요한 feature 일 수 있다고 판단을 내릴 수는 있습니다.
# 이제 모델이 학습하지 않았던(보지 않았던) 테스트셋을 모델에 주어서, 생존여부를 예측해보겠습니다.



# 캐글에서 준 파일, gender_submission.csv 파일을 읽어서 제출 준비를 하겠습니다.
submission = pd.read_csv('../input/sample_submission.csv')
submission.head()
# 이제 testset 에 대하여 예측을 하고, 결과를 csv 파일로 저장해보겠습니다.
prediction = model.predict(X_test)

submission['Survived'] = prediction
submission.to_csv('./my_first_submission.csv', index=False)