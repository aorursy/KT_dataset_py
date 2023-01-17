# data analysis 

import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt



# hide useless warnings...

import warnings

warnings.filterwarnings('ignore')



# import data to pandas instance from csv file

train_df = pd.read_csv('../input/train.csv')  # training dataframe

test_df  = pd.read_csv('../input/test.csv')   # test dataframe

train_df.head()
# There is 891 observations and 12 variables 

train_df.shape

train_df.describe()
train_df.isnull().sum()
f,ax=plt.subplots(1,2,figsize=(18,8))

train_df['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=train_df,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

train_df['Survived'][train_df['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

train_df['Survived'][train_df['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[0].set_title('Survived (male)')

ax[1].set_title('Survived (female)')



plt.show()
# 549 died, 342 survived in train data. Proportion is 61.6162% | 38.3838%

print(train_df['Sex'].value_counts(normalize=True))

train_df.groupby(['Sex','Survived'])['Survived'].count()
pd.crosstab([train_df['Sex'],train_df['Survived']],train_df['Pclass'],margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

train_df['Pclass'].value_counts().plot.pie(explode=[0,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Number Of Passengers By Pclass')

sns.countplot('Pclass',hue='Survived',data=train_df,ax=ax[1])

ax[1].set_title('Survived vs Dead by Pclass')

plt.show()
sns.factorplot('Pclass','Survived',hue='Sex',data=train_df)

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("Sex","Age", hue="Survived", data=train_df,split=True,ax=ax[0])

ax[0].set_title('Sex and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Pclass","Age", hue="Survived", data=train_df,split=True,ax=ax[1])

ax[1].set_title('Pclass and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
f,ax=plt.subplots(1,2,figsize=(20,10))

train_df[train_df['Survived']==0]['Age'].plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')

ax[0].set_title('Survived= 0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)

train_df[train_df['Survived']==1]['Age'].plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')

ax[1].set_title('Survived= 1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)

plt.show()
pd.crosstab([train_df['Embarked'],train_df['Pclass']],[train_df['Sex'],train_df['Survived']],margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(2,2,figsize=(20,15))

sns.countplot('Embarked',data=train_df,ax=ax[0,0])

ax[0,0].set_title('No. Of Passengers Boarded')

sns.countplot('Embarked',hue='Sex',data=train_df,ax=ax[0,1])

ax[0,1].set_title('Male-Female Split for Embarked')

sns.countplot('Embarked',hue='Survived',data=train_df,ax=ax[1,0])

ax[1,0].set_title('Embarked vs Survived')

sns.countplot('Embarked',hue='Pclass',data=train_df,ax=ax[1,1])

ax[1,1].set_title('Embarked vs Pclass')

# plt.subplots_adjust(wspace=0.2,hspace=0.5)

plt.show()
train_df['Embarked'].fillna('S',inplace=True)

test_df['Embarked'].fillna('S',inplace=True)
# reform all 'string' data to 'integer'

train_df['Sex'].replace(['male','female'],[0,1],inplace=True)

test_df['Sex'].replace(['male','female'],[0,1],inplace=True)



train_df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

test_df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure





## prediction with logistic regression

train, test = train_test_split(train_df,test_size=0.3,random_state=0)

target_col = ['Pclass', 'Sex', 'Embarked']



train_X=train[target_col]

train_Y=train['Survived']

test_X=test[target_col]

test_Y=test['Survived']



features_one = train_X.values

target = train_Y.values
tree_model = DecisionTreeClassifier()

tree_model.fit(features_one, target)

dt_prediction = tree_model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(dt_prediction, test_Y))



# test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

test_features = test_df[target_col].values



# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId =np.array(test_df["PassengerId"]).astype(int)

# my_prediction = my_tree_one.predict(test_features)

dt_prediction_result = tree_model.predict(test_features)

dt_solution = pd.DataFrame(dt_prediction_result, PassengerId, columns = ["Survived"])

# print(my_solution)



# Check that your data frame has 418 entries

print(dt_solution.shape)



# Write your solution to a csv file with the name my_solution.csv

dt_solution.to_csv("my_solution_two.csv", index_label = ["PassengerId"]) 
train_df['Title'] = train_df['Name'].str.extract('([A-Za-z]+)\.')

pd.crosstab(train_df['Title'], train_df['Sex'])
train_df['Title'] = train_df['Title'].replace(['Capt', 'Col',

'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')



train_df['Title'] = train_df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

train_df['Title'] = train_df['Title'].replace(['Mlle', 'Ms'], 'Miss')

train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')



train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# pd.crosstab(train_df['Title'], train_df['Sex'])
train_df.groupby('Title')['Age'].mean()
train_df.loc[(train_df['Age'].isnull())&(train_df['Title']=='Master'),'Age'] = 5

train_df.loc[(train_df['Age'].isnull())&(train_df['Title']=='Miss'),'Age'] = 22

train_df.loc[(train_df['Age'].isnull())&(train_df['Title']=='Mr'),'Age'] = 33

train_df.loc[(train_df['Age'].isnull())&(train_df['Title']=='Mrs'),'Age'] = 36

train_df.loc[(train_df['Age'].isnull())&(train_df['Title']=='Rare'),'Age'] = 45

train_df.loc[(train_df['Age'].isnull())&(train_df['Title']=='Royal'),'Age'] = 43
f,ax=plt.subplots(1,2,figsize=(20,10))

train_df[train_df['Survived']==0]['Age'].plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')

ax[0].set_title('Survived= 0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)

train_df[train_df['Survived']==1]['Age'].plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')

ax[1].set_title('Survived= 1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)

plt.show()
train_df['AgeGroup'] = 0

train_df.loc[ train_df['Age'] <= 7, 'AgeGroup'] = 0

train_df.loc[(train_df['Age'] > 7) & (train_df['Age'] <= 18), 'AgeGroup'] = 1

train_df.loc[(train_df['Age'] > 18) & (train_df['Age'] <= 30), 'AgeGroup'] = 2

train_df.loc[(train_df['Age'] > 30) & (train_df['Age'] <= 40), 'AgeGroup'] = 3

train_df.loc[(train_df['Age'] > 40) & (train_df['Age'] <= 60), 'AgeGroup'] = 4

train_df.loc[ train_df['Age'] > 60, 'AgeGroup'] = 5

pd.crosstab(train_df['AgeGroup'], train_df['Survived'])
f,ax=plt.subplots(1,1,figsize=(10,10))

sns.countplot('AgeGroup',hue='Survived',data=train_df, ax=ax)

plt.show()
train_df['TitleKey'] = 0

title_mapping = {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rare': 4, 'Royal': 5}

train_df['TitleKey'] = train_df['Title'].map(title_mapping)
train_df['FamilyMembers'] = train_df['SibSp'] + train_df['Parch'] + 1

pd.crosstab([train_df['FamilyMembers']],train_df['Survived']).style.background_gradient(cmap='summer_r')
train_df['IsAlone'] = 0

train_df.loc[train_df['FamilyMembers'] == 1, 'IsAlone'] = 1



f, ax=plt.subplots(1,2,figsize=(20,8))

# sns.barplot('FamilyMembers','Survived',data=train_df,ax=ax[0])

ax[0].set_title('Survived, with family')

train_df['Survived'][train_df['IsAlone']==0].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

train_df['Survived'][train_df['IsAlone']==1].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)

# sns.barplot('IsAlone','Survived',data=train_df,ax=ax[1])

ax[1].set_title('IsAlone Survived')

plt.show()
sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
# Do the same things to test data

test_df['Title'] = test_df['Name'].str.extract('([A-Za-z]+)\.')

test_df['Title'] = test_df['Title'].replace(['Capt', 'Col',

'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')



test_df['Title'] = test_df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

test_df['Title'] = test_df['Title'].replace(['Mlle', 'Ms', 'Lady'], 'Miss')

test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')



test_df.loc[(test_df['Age'].isnull())&(test_df['Title']=='Master'),'Age'] = 5

test_df.loc[(test_df['Age'].isnull())&(test_df['Title']=='Miss'),'Age'] = 22

test_df.loc[(test_df['Age'].isnull())&(test_df['Title']=='Mr'),'Age'] = 33

test_df.loc[(test_df['Age'].isnull())&(test_df['Title']=='Mrs'),'Age'] = 36

test_df.loc[(test_df['Age'].isnull())&(test_df['Title']=='Rare'),'Age'] = 45

test_df.loc[(test_df['Age'].isnull())&(test_df['Title']=='Royal'),'Age'] = 43



test_df['TitleKey'] = 0

title_mapping = {'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rare': 4, 'Royal': 5}

test_df['TitleKey'] = train_df['Title'].map(title_mapping)



test_df['AgeGroup'] = 0

test_df.loc[ test_df['Age'] <= 7, 'AgeGroup'] = 0

test_df.loc[(test_df['Age'] > 7) & (test_df['Age'] <= 18), 'AgeGroup'] = 1

test_df.loc[(test_df['Age'] > 18) & (test_df['Age'] <= 30), 'AgeGroup'] = 2

test_df.loc[(test_df['Age'] > 30) & (test_df['Age'] <= 40), 'AgeGroup'] = 3

test_df.loc[(test_df['Age'] > 40) & (test_df['Age'] <= 60), 'AgeGroup'] = 4

test_df.loc[ test_df['Age'] > 60, 'AgeGroup'] = 5



test_df['FamilyMembers'] = test_df['SibSp'] + test_df['Parch'] + 1

test_df['IsAlone'] = 0

test_df.loc[test_df['FamilyMembers'] == 1, 'IsAlone'] = 1
train, test = train_test_split(train_df,test_size=0.3,random_state=0)

target_col = ['Pclass', 'Sex', 'Embarked', 'TitleKey', 'AgeGroup', 'IsAlone']



train_X=train[target_col]

train_Y=train['Survived']

test_X=test[target_col]

test_Y=test['Survived']



features_one = train_X.values

target = train_Y.values



tree_model = DecisionTreeClassifier()

tree_model.fit(features_one, target)

dt_prediction = tree_model.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(dt_prediction, test_Y))



# test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

test_features = test_df[target_col].values



# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId =np.array(test_df["PassengerId"]).astype(int)

# my_prediction = my_tree_one.predict(test_features)

dt_prediction_result = tree_model.predict(test_features)

dt_solution = pd.DataFrame(dt_prediction_result, PassengerId, columns = ["Survived"])

# print(my_solution)



# Check that your data frame has 418 entries

print(dt_solution.shape)



# Write your solution to a csv file with the name my_solution.csv

dt_solution.to_csv("my_solution_three.csv", index_label = ["PassengerId"]) 
f,ax=plt.subplots(1,3,figsize=(20,8))

sns.distplot(train_df[train_df['Pclass']==1]['Fare'],ax=ax[0])

ax[0].set_title('Fares in Pclass 1')

sns.distplot(train_df[train_df['Pclass']==2]['Fare'],ax=ax[1])

ax[1].set_title('Fares in Pclass 2')

sns.distplot(train_df[train_df['Pclass']==3]['Fare'],ax=ax[2])

ax[2].set_title('Fares in Pclass 3')

plt.show()