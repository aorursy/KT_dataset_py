import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn')
sns.set(font_scale=2.5) 
# 이 두줄은 본 필자가 항상 쓰는 방법입니다. 
# matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.
import missingno as msno


%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.describe()
# describe()를 쓰면 각 feature 가 가진 통계치들을 반환해줍니다.

test.describe()

train.isnull().sum()/len(train)*100
test.isnull().sum()/len(train)*100
msno.matrix(df=train, figsize=(8, 8), color=(0.8, 0.5, 0.2))

msno.matrix(df=test, figsize=(8, 8), color=(0.8, 0.5, 0.2))

train.head()
f, ax = plt.subplots(1, 2, figsize=(18, 8))

train['Survived'].value_counts().plot.pie( autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Pie plot - Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=train, ax=ax[1])
ax[1].set_title('Count plot - Survived')

plt.show()
train.groupby(['Pclass','Survived'])['Pclass'].count()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=train, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
plt.show()
f, ax = plt.subplots(1, 2, figsize=(18, 8))
train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=train, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()

train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(train['Sex'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass', 'Survived', hue='Sex', data=train, size=6, aspect=1.5)
sns.factorplot(x='Sex', y='Survived', col='Pclass',
               data=train, satureation=.5,
               size=9, aspect=1
              )
print('제일 나이 많은 탑승객 : ',round(train['Age'].max()))
print('제일 어린 탑승객 : ',round(train['Age'].min()))
print('탑승객 평균 나이 : ',round(train['Age'].mean()))
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(train[train['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(train[train['Survived'] == 0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'Survived == 0'])
plt.show()
# Age distribution withing classes
plt.figure(figsize=(8, 6))
train['Age'][train['Pclass'] == 1].plot(kind='kde')
train['Age'][train['Pclass'] == 2].plot(kind='kde')
train['Age'][train['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])
cummulate_survival_ratio = []
for i in range(1, 80):
    cummulate_survival_ratio.append(train[train['Age'] < i]['Survived'].sum() / len(train[train['Age'] < i]['Survived']))
    
plt.figure(figsize=(7, 7))
plt.plot(cummulate_survival_ratio)
plt.title('Survival rate change depending on range of Age', y=1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~x)')
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

#=== Pclass - Age 간 생존 여부 ===#
sns.violinplot("Pclass","Age", hue="Survived", data=train, scale='count', split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))

#=== Sex - Age간 생존 여부 ===#
sns.violinplot("Sex","Age", hue="Survived", data=train, scale='count', split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()
f, ax = plt.subplots(1, 1, figsize=(7, 7))
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
f,ax=plt.subplots(2, 2, figsize=(20,15))
sns.countplot('Embarked', data=train, ax=ax[0,0])
ax[0,0].set_title('(1) No. Of Passengers Boarded')

sns.countplot('Embarked', hue='Sex', data=train, ax=ax[0,1])
ax[0,1].set_title('(2) Male-Female Split for Embarked')

sns.countplot('Embarked', hue='Survived', data=train, ax=ax[1,0])
ax[1,0].set_title('(3) Embarked vs Survived')

sns.countplot('Embarked', hue='Pclass', data=train, ax=ax[1,1])
ax[1,1].set_title('(4) Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

print("Maximum size of Family: ", train['FamilySize'].max())
print("Minimum size of Family: ", train['FamilySize'].min())
f,ax=plt.subplots(1, 3, figsize=(40,10))
sns.countplot('FamilySize', data=train, ax=ax[0])
ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)

sns.countplot('FamilySize', hue='Survived', data=train, ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(train['Fare'], color='b', label='Skewness : {:.2f}'.format(train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')
train.isnull().sum()
test.isnull().sum()
test.loc[test.Fare.isnull(), 'Fare'] = test['Fare'].mean() # testset 에 있는 nan value 를 평균값으로 치환합니다.
test.isnull().sum()
print(train['Fare'].min(), test['Fare'].min())
train['Fare'] = np.log1p(train['Fare'])
test['Fare'] = np.log1p(test['Fare'])
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
g = sns.distplot(train['Fare'], color='b', label='Skewness : {:.2f}'.format(train['Fare'].skew()), ax=ax)
g = g.legend(loc='best')
train.isnull().sum()/len(train)
# Train의 Null 개수에 전체 Row 개수를 나눠줌
train = train.drop(['Cabin'],1)
test = test.drop(['Cabin'],1)
# Cabin Column을 Drop시킴
train.isnull().sum()/len(train)
# Train의 Null 개수에 전체 Row 개수를 나눠줌
# Cabin이 사라진 것을 확인
train.isnull().sum()/len(train)
# Ticket 값은 Null 값이 없음
train.head()
# 다만 글자로 되어있어서 변환에 고민이 됨
train['Ticket'].value_counts()
# Ticket의 종류별로 얼마나 있는지 확인 (자주 사용되는 함수)
train.isnull().sum()
train['Name'].head()
train['Initial'] = train['Name'].apply(lambda x: x.split(',')[1].strip())
test['Initial'] = test['Name'].apply(lambda x: x.split(',')[1].strip())

train['Initial'].head()
train['Initial'] = train['Initial'].apply(lambda x: x.split('.')[0].strip())
test['Initial'] = test['Initial'].apply(lambda x: x.split('.')[0].strip())

train['Initial'].head()
train.groupby(['Sex','Initial'])['Initial'].count()
train['Initial'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms','Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess','Jonkheer'],
                         ['Mr', 'Mrs', 'Miss', 'Master',  'Mr' ,'other','Mr', 'Miss', 'Miss', 'Mr', 'Mrs', 'Mr', 'Miss', 'Other', 'Mr' , 'Mrs', 'Other'])
train['Initial'] = train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                                            ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'])
test['Initial'] = test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                                          ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'])
train['Initial'].unique()
test['Initial'].unique()
train.groupby(['Sex','Initial'])['Initial'].count()
train.groupby('Initial').mean()
train.groupby('Initial')['Survived'].mean().plot.bar()
train.isnull().sum()
train.groupby('Initial').mean()
train.loc[(train['Age'].isnull())&(train['Initial']=='Mr'),'Age'] = 33
train.loc[(train['Age'].isnull())&(train['Initial']=='Mrs'),'Age'] = 36
train.loc[(train['Age'].isnull())&(train['Initial']=='Master'),'Age'] = 5
train.loc[(train['Age'].isnull())&(train['Initial']=='Miss'),'Age'] = 22
train.loc[(train['Age'].isnull())&(train['Initial']=='Other'),'Age'] = 46

test.loc[(test['Age'].isnull())&(test['Initial']=='Mr'),'Age'] = 33
test.loc[(test['Age'].isnull())&(test['Initial']=='Mrs'),'Age'] = 36
test.loc[(test['Age'].isnull())&(test['Initial']=='Master'),'Age'] = 5
test.loc[(test['Age'].isnull())&(test['Initial']=='Miss'),'Age'] = 22
test.loc[(test['Age'].isnull())&(test['Initial']=='Other'),'Age'] = 46
train['Embarked'].isnull().sum()
train['Embarked'].value_counts()
train['Embarked'].fillna('S', inplace=True)
train['Embarked'].isnull().sum()
train['Age_cat'] = 0
train.loc[train['Age'] < 10, 'Age_cat'] = 0   # Age가 10보다 작을때는 0
train.loc[(10 <= train['Age']) & (train['Age'] < 20), 'Age_cat'] = 1   # Age가 10~20일때는 1
train.loc[(20 <= train['Age']) & (train['Age'] < 30), 'Age_cat'] = 2   # Age가 20~30일때는 2
train.loc[(30 <= train['Age']) & (train['Age'] < 40), 'Age_cat'] = 3   # Age가 30~40일때는 3
train.loc[(40 <= train['Age']) & (train['Age'] < 50), 'Age_cat'] = 4   # Age가 40~50일때는 4
train.loc[(50 <= train['Age']) & (train['Age'] < 60), 'Age_cat'] = 5   # Age가 50~60일때는 5
train.loc[(60 <= train['Age']) & (train['Age'] < 70), 'Age_cat'] = 6   # Age가 60~70일때는 6
train.loc[70 <= train['Age'], 'Age_cat'] = 7   # Age가 70이상 일때는 7 

test['Age_cat'] = 0   # 위와 동일함, Test도 똑같이 적용해야 함
test.loc[test['Age'] < 10, 'Age_cat'] = 0
test.loc[(10 <= test['Age']) & (test['Age'] < 20), 'Age_cat'] = 1
test.loc[(20 <= test['Age']) & (test['Age'] < 30), 'Age_cat'] = 2
test.loc[(30 <= test['Age']) & (test['Age'] < 40), 'Age_cat'] = 3
test.loc[(40 <= test['Age']) & (test['Age'] < 50), 'Age_cat'] = 4
test.loc[(50 <= test['Age']) & (test['Age'] < 60), 'Age_cat'] = 5
test.loc[(60 <= test['Age']) & (test['Age'] < 70), 'Age_cat'] = 6
test.loc[70 <= test['Age'], 'Age_cat'] = 7
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


print(category_age(10))
print(category_age(33))
print(category_age(42))
print(category_age(45))
print(category_age(47))
print(category_age(55))
print(category_age(85))

train['Age_cat'] = train['Age'].apply(category_age)
train[['Age','Age_cat']].head()
train['Initial'] = train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
train['Initial'] = test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
train['Embarked'].isnull().sum()
test['Embarked'].isnull().sum()
train['Embarked'].unique()
train['Embarked'].value_counts()
train['Embarked'] = train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test['Embarked'] = test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
train['Embarked'].unique()
train['Sex'].isnull().sum()
test['Sex'].isnull().sum()
train['Sex'].unique()
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})
train['Sex'].unique()
test['Sex'].unique()
heatmap_data = train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']] 
# train의 'Survived', 'Pclass', 'Sex'... Column들을 heatmap_data로 정해줍니다.
# 여러 열을 선택할때 대괄호 두번을 사용하면됩니다. --> [[ ]]

colormap = plt.cm.RdBu
plt.figure(figsize=(10, 8))
plt.title('Pearson Correlation of Features', y=1.05, size=13)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,     # 앞에서 지정해준 heatmap_data를 넣어줍니다.
           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})

#=== 위의 설명의 예시입니다. ===#
t = [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
pd.DataFrame(t,columns=['Initial_Master','Initial_Miss','Initial_Mr','Initial_Mrs','Initial_Other'])
train = pd.get_dummies(train, columns=['Initial'], prefix='Initial')
test = pd.get_dummies(test, columns=['Initial'], prefix='Initial')
train.head()
train = pd.get_dummies(train, columns=['Embarked'], prefix='Embarked')
test = pd.get_dummies(test, columns=['Embarked'], prefix='Embarked')
train.head()
print(train.shape, test.shape)
train = train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1)
test = test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket'], axis=1)
print(train.shape, test.shape)
train.head()
test.head()
from sklearn.ensemble import RandomForestClassifier # 유명한 randomforestclassfier 입니다. 
from sklearn import metrics # 모델의 평가를 위해서 씁니다
from sklearn.model_selection import train_test_split # traning set을 쉽게 나눠주는 함수입니다.
train['Survived'].unique()
y = train['Survived']
X = train.drop(['Survived'],1)
X_test = test
print(train.shape, test.shape, "-->", X.shape,y.shape,X_test.shape)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X,y,random_state=30, test_size =0.3)
# 이런 양식에 넣으면 알아서 나눠짐, test_size에 비율을 넣으면 그만큼 Validation으로 나눠짐
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier()

from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression()

from lightgbm import LGBMClassifier
model3 = LGBMClassifier()

for model in [model1, model2, model3]:
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    print('train_score : ', (pred_train == y_train).mean())
    pred_valid = model.predict(X_valid)
    print('valid_score : ', (pred_valid == y_valid).mean())
    print("==============================================")
y_train
pred_train = model.predict(X_train)
(pred_train == y_train).mean()
pred_train = model.predict(X_valid)
(pred_train == y_valid).mean()
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission.head()
prediction = model.predict(X_test)
prediction
submission['Survived'] = prediction
submission.head()
submission.to_csv('./final_submission.csv', index=False)
from pandas import Series

feature_importance = model.feature_importances_
Series_feat_imp = Series(feature_importance, index=test.columns)
plt.figure(figsize=(8, 8))
Series_feat_imp.sort_values(ascending=True).plot.barh()
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
