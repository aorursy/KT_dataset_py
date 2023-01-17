import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')

sns.set(font_scale=2.5) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.

import missingno as msno



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
train.describe()
train.shape
train.columns
for col in train.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (

        train[col].isnull().sum() / train[col].shape[0]))

    print(msg)
for col in test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (test[col].isnull().sum() / test[col].shape[0]))

    print(msg)
msno.matrix(df=train.iloc[:,:], figsize=(8,8), color=(0.8, 0.5, 0.2))
msno.bar(df=train.iloc[:, :], figsize=(8, 8), color=(0.8, 0.5, 0.2))
f, ax = plt.subplots(1, 2, figsize=(18, 8))



train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=train, ax=ax[1])

ax[1].set_title('Count plot - Survived')



plt.show()
train['Survived'].value_counts().plot.pie(explode=[0, 0.1])
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
pd.crosstab(train['Pclass'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
pd.crosstab(train['Pclass'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar()
y_position = 1.02

f, ax = plt.subplots(1, 2, figsize=(18, 8))

train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])

ax[0].set_title('Number of Passengers By Pclass', y=y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()
y_position = 1.02

f, ax = plt.subplots(1, 2, figsize=(18, 8))

train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])

ax[0].set_title('Number of Passengers By Pclass', y=y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()

f, ax = plt.subplots(1,2, figsize = (18,8))

train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
f, ax = plt.subplots(1,2, figsize = (18,8))

train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab(train['Sex'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass', 'Survived', hue='Sex', data=train, size=6, aspect=1.5)
sns.factorplot(x='Sex', y='Survived', col='Pclass', data=train, satureation=.5, size=9, aspect=1)
print('제일 나이 많은 탑승객: {:.1f}Years'.format(train['Age'].max()))

print('제일 어린 탑승객: {:.1f}Years'.format(train['Age'].min()))

print('탑승객 평균 나이: {:.1f}Years'.format(train['Age'].mean()))
fig, ax = plt.subplots(1,1,figsize=(9,5))

sns.kdeplot(train[train['Survived']==1]['Age'], ax=ax)

sns.kdeplot(train[train['Survived']==0]['Age'], ax=ax)

plt.legend(['Survived==1', 'Survived==0'])

plt.show()
plt.figure(figsize=(8,6))

train['Age'][train['Pclass']==1].plot(kind='kde')

train['Age'][train['Pclass']==2].plot(kind='kde')

train['Age'][train['Pclass']==3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])
cummulate_survival_ratio = []

for i in range(1, 80):

    cummulate_survival_ratio.append(train[train['Age'] < i]['Survived'].sum() / len(train[train['Age'] < i]['Survived']))

    

plt.figure(figsize=(7,7))

plt.plot(cummulate_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Pclass","Age", hue="Survived", data=train, scale='count', split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data=train, scale='count', split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
f, ax = plt.subplots(1,1,figsize=(7,7))

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)

f, ax=plt.subplots(2,2,figsize=(20,15))

sns.countplot('Embarked', data=train, ax=ax[0,0])

ax[0,0].set_title('(1) No. Of Passengers Boarded')

sns.countplot('Embarked', hue='Sex', data=train, ax=ax[0,1])

ax[0,1].set_title('(2) Male-Female Split for Embarked')

sns.countplot('Embarked', hue='Survived', data=train, ax=ax[1,0])

ax[1,0].set_title('(3) Embarked vs Survived')

sns.countplot('Embarked', hue='Pclass', data=train, ax=ax[1,1])

ax[1,1].set_title('(4) Embarked vs Pclass')

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show
train['FamilySize']= train['SibSp'] + train['Parch']+1

test['FamilySize'] = test['SibSp'] + test['Parch']+1

print("Maximum size of Family: ", train['FamilySize'].max())

print("Minimum size of Family: ", train['FamilySize'].min())
f, ax = plt.subplots(1,3, figsize=(40,10))

sns.countplot('FamilySize', data=train, ax=ax[0])

ax[0].set_title('(1) No. Of Passengers Boarded')
f, ax = plt.subplots(1,3, figsize=(40,10))

sns.countplot('FamilySize', data=train, ax=ax[0])

ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)



sns.countplot('FamilySize', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('(2) Survived countplot depending on FamilySize', y=1.02)



train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize', y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
fig, ax= plt.subplots(1,1, figsize=(8,8))

g = sns.distplot(train['Fare'], color='b', label='Skewness : {:.2f}'.format(train['Fare'].skew()),ax=ax)

g=g.legend(loc='best')
test.loc[test.Fare.isnull(), 'Fare'] = test['Fare'].mean()



train['Fare'] = train['Fare'].map(lambda i: np.log(i) if i>0 else 0)

test['Fare'] = test['Fare'].map(lambda i: np.log(i) if i>0 else 0)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(train['Fare'], color='b', label='Skewness : {:.2f}'.format(train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
train.head()
train['Ticket'].value_counts()