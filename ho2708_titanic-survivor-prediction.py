# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
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
train = pd.read_csv("../input/2019-1st-ml-month-with-kakr/train.csv")

test = pd.read_csv("../input/2019-1st-ml-month-with-kakr/test.csv")
from IPython.display import Image

Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")
Image(url= "https://upload.wikimedia.org/wikipedia/commons/8/84/Titanic_cutaway_diagram.png")
train.head()
test.head()
train.describe()
test.describe()
train.info()
test.info()
print(train.isnull().sum())

train.isnull().sum() / train.shape[0]
print(test.isnull().sum())

test.isnull().sum() / test.shape[0]
f, ax = plt.subplots(1, 2, figsize=(18, 8))



train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=train, ax=ax[1])

ax[1].set_title('Count plot - Survived')



plt.show()
pd.crosstab(train['Pclass'], train['Survived'], margins=True).style.background_gradient(cmap='autumn_r')
# Explore Pclass vs Survived

g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
y_position = 1.02

fig, ax = plt.subplots(1, 2, figsize = (18, 8))

train['Pclass'].value_counts().plot.bar(color = ['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])

ax[0].set_title('Number of Passengers By Pclass', y = y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass', hue = 'Survived', data = train, ax = ax[1])

ax[1].set_title('Pclass: Survived vs Dead', y = y_position)

plt.show()
pd.crosstab(train['Sex'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
fig, ax = plt.subplots(1, 2, figsize = (18, 8))

train[['Sex', 'Survived']].groupby(['Sex'], as_index = True).mean().plot.bar(ax = ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue = 'Survived', data = train, ax = ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
fig, ax = plt.subplots(1, figsize = (18, 8))

sns.countplot('Pclass', hue = 'Sex', data = train)

ax.set_title('Pclass - Sex')

plt.show()
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(train['Age'].max()))

print('제일 어린 탑승객 : {:.1f} Years'.format(train['Age'].min()))

print('탑승객 평균 나이 : {:.1f} Years'.format(train['Age'].mean()))
pd.crosstab(train['Age'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
# Explore Age vs Survived

g = sns.FacetGrid(train, col='Survived')

g = g.map(sns.distplot, "Age")
fig, ax = plt.subplots(1, 1, figsize = (9, 5))

sns.kdeplot(train[train['Survived'] == 1]['Age'], ax = ax)

sns.kdeplot(train[train['Survived'] == 0]['Age'], ax = ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
plt.figure(figsize = (8, 6))

train['Age'][train['Pclass'] == 1].plot(kind = 'kde')

train['Age'][train['Pclass'] == 2].plot(kind = 'kde')

train['Age'][train['Pclass'] == 3].plot(kind = 'kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])
fig, ax = plt.subplots(1, 2, figsize = (18, 8))

sns.violinplot("Pclass", "Age", hue = "Survived", data = train, scale = 'count', split = True, ax = ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0, 110, 19))

sns.violinplot("Sex", "Age", hue = "Survived", data = train, scale = 'count', split = True, ax = ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0, 110, 19))

plt.show()
pd.crosstab(train['Embarked'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
fig, ax = plt.subplots(1, 1, figsize = (7, 7))

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index = True).mean().sort_values(by = 'Survived', ascending = False).plot.bar(ax = ax)
fig, ax = plt.subplots(2, 2, figsize = (20, 15))

sns.countplot('Embarked', data = train, ax = ax[0, 0])

ax[0, 0].set_title('(1) No. Of passengers Boarded')

sns.countplot('Embarked', hue = 'Sex', data = train, ax = ax[0, 1])

ax[0, 1].set_title('(2) Male-Female Split for Embarked')

sns.countplot('Embarked', hue = 'Survived', data = train, ax = ax[1,0])

ax[1, 0].set_title('(3) Embarked vs Survived')

sns.countplot('Embarked', hue = 'Pclass', data = train, ax = ax[1, 1])

ax[1, 1].set_title('(4) Embarked vs Pclass')

plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

plt.show()
fig, ax = plt.subplots(1, figsize = (18, 8))

sns.violinplot("Embarked", "Age", hue = "Survived", data = train, scale = 'count', split = True)

ax.set_title('Age vs Embarked')

ax.set_yticks(range(0, 110, 19))

plt.show()
pd.crosstab(train['SibSp'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
fig, ax = plt.subplots(2, 3, figsize = (40, 20))

sns.countplot('SibSp', data = train, ax = ax[0, 0])

ax[0, 0].set_title('(1) No. Of passengers Boarded', y = 1.02)



sns.countplot('SibSp', hue = 'Survived', data = train, ax = ax[0, 1])

ax[0, 1].set_title('(2) Survived countplot depending on SibSp', y = 1.02)



train[['SibSp', 'Survived']].groupby(['SibSp'], as_index = True).mean().sort_values(by = 'Survived', ascending = False).plot.bar(ax = ax[0, 2])

ax[0, 2].set_title('(3) Survived rate depending on SibSp', y = 1.02)



sns.countplot('SibSp', hue = 'Pclass', data = train, ax = ax[1, 0])

ax[1, 0].set_title('(4) SibSp vs Pclass')



sns.countplot('SibSp', hue = 'Sex', data = train, ax = ax[1, 1])

ax[1, 1].set_title('(2) Male-Female Split for SibSp')



plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

plt.show()
pd.crosstab(train['Parch'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
fig, ax = plt.subplots(2, 3, figsize = (40, 20))

sns.countplot('Parch', data = train, ax = ax[0, 0])

ax[0, 0].set_title('(1) No. Of passengers Boarded', y = 1.02)



sns.countplot('Parch', hue = 'Survived', data = train, ax = ax[0, 1])

ax[0, 1].set_title('(2) Survived countplot depending on Parch', y = 1.02)



train[['Parch', 'Survived']].groupby(['Parch'], as_index = True).mean().sort_values(by = 'Survived', ascending = False).plot.bar(ax = ax[0, 2])

ax[0, 2].set_title('(3) Survived rate depending on Parch', y = 1.02)



sns.countplot('Parch', hue = 'Pclass', data = train, ax = ax[1, 0])

ax[1, 0].set_title('(4) Parch vs Pclass')



sns.countplot('Parch', hue = 'Sex', data = train, ax = ax[1, 1])

ax[1, 1].set_title('(2) Male-Female Split for Parch')



plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

plt.show()
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
pd.crosstab(train['FamilySize'], train['Survived'], margins=True).style.background_gradient(cmap='summer_r')
fig, ax = plt.subplots(2, 3, figsize = (40, 20))

sns.countplot('FamilySize', data = train, ax = ax[0, 0])

ax[0, 0].set_title('(1) No. Of passengers Boarded', y = 1.02)



sns.countplot('FamilySize', hue = 'Survived', data = train, ax = ax[0, 1])

ax[0, 1].set_title('(2) Survived countplot depending on FamilySize', y = 1.02)



train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = True).mean().sort_values(by = 'Survived', ascending = False).plot.bar(ax = ax[0, 2])

ax[0, 2].set_title('(3) Survived rate depending on FamilySize', y = 1.02)



sns.countplot('FamilySize', hue = 'Pclass', data = train, ax = ax[1, 0])

ax[1, 0].set_title('(4) FamilySize vs Pclass')



sns.countplot('FamilySize', hue = 'Sex', data = train, ax = ax[1, 1])

ax[1, 1].set_title('(2) Male-Female Split for FamilySize')



plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(train['Fare'], color='b', label='Skewness : {:.2f}'.format(train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
train['Ticket'].value_counts()
train.head()
train['Title']= train.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

test['Title']= test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
pd.crosstab(train['Title'], train['Sex']).T.style.background_gradient(cmap='summer_r') 
train.groupby('Title').mean()
train['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)



test['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
def Is_Alone(x):

    if x == 1:

        return 0

    else:

        return 1

    

train['IsAlone'] = train['FamilySize'].apply(Is_Alone)

test['IsAlone'] = test['FamilySize'].apply(Is_Alone)
train['FamilyName']= train.Name.str.extract('([A-Za-z]+)\,') #lets extract the Salutations

test['FamilyName']= test.Name.str.extract('([A-Za-z]+)\,') #lets extract the Salutations
train['FamilyName2']= train.Name.str.extract('([A-Za-z]+)\)') #lets extract the Salutations

test['FamilyName2']= test.Name.str.extract('([A-Za-z]+)\)') #lets extract the Salutations
if train.Ticket.str.extract('(\d+)').size > 1:

    train['Ticket2']= train.Ticket.str.extract(' (\d+)') #lets extract the Salutations



if test.Ticket.str.extract('(\d+)').size > 1:

    test['Ticket2']= test.Ticket.str.extract(' (\d+)') #lets extract the Salutations



train["Ticket2"] = train["Ticket2"].fillna(train["Ticket"])



test["Ticket2"] = test["Ticket2"].fillna(test["Ticket"])
train.head()
EDA_train = pd.read_csv("../input/titanic-eda-last/titanic_EDA_train.csv")

EDA_test = pd.read_csv("../input/titanic-eda-last/titanic_EDA_test.csv")

EDA_all = pd.read_csv("../input/titanic-eda-last/titanic_EDA_all - .csv")
EDA_train.head()
EDA_test.head()
EDA_all.head()
FE_all = pd.concat([train, test])



FE_all.head()

#다시 구분할 때 사용

#df_test = all_df.loc[all_df['price'].isnull()]

#df_train = all_df.loc[all_df['price'].notnull()]
#DataFrame(data, columns=['year', 'state', ‘pop'])

#pd(FE_all, columns=['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','FamilySize','Title','IsAlone','FamilyName','FamilyName2'])



cols = FE_all.columns.tolist()



FE_all = FE_all[['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Ticket2','Fare','Cabin','Embarked','FamilySize','Title','IsAlone','FamilyName','FamilyName2']]



FE_all.head()
pd.set_option('display.max_rows', 7000)



#가족이 있는 사람들의 Title별

train[train['IsAlone'] == 1].sort_values(by=['Cabin', 'Ticket2', 'Fare', 'FamilySize']).groupby('Title').mean()
FE_all[(FE_all['Title'] == "Mrs") & (FE_all['SibSp'] == 1) & (FE_all['Parch'] > 1)]
FE_all[(FE_all['Title'] == "Mr") & (FE_all['SibSp'] == 1) & (FE_all['Parch'] > 1)]
FE_all[FE_all['FamilySize'] == 7].sort_values(by=['FamilySize', 'FamilyName','Ticket','Fare','Age'])
#아래 옵션을 통해 데이터 프레임의 전체를 출력하여 볼 수 있다.

#pd.set_option('display.max_rows', 7000)

train[train['FamilySize'] == 2].sort_values(by=['FamilySize', 'FamilyName','Ticket2','Fare','Age'])