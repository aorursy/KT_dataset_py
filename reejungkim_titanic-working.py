# Load in libraries



import warnings

warnings.filterwarnings('ignore')



#libraries for handling data

import pandas as pd

import numpy as np



#libraries for data visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



#libaries for modelling

# Modelling Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier



# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.cross_validation import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV 
#import and read head of train data set

train = pd.read_csv('../input/train.csv')

train.head(3)
#import and read head of test data set

test = pd.read_csv('../input/test.csv')

test.head(3)
#import and read head of gender_submission data set

genderSub = pd.read_csv('../input/gender_submission.csv')

genderSub.head(3)
#append the train and test data sets and name it full

#full = train.append( test , ignore_index = True )

#train = full[ :891 ]



#print ('Datasets:' , 'full:' , full.shape , 'titanic:' , train.shape)
#merging test and train data sets

#dataFull = train.append(test, ignore_index= True)

#dataFull.head()
train.info()
train.isna().sum()

#train.isnull().sum()

#train['Age'].isnull().sum()
test.info()
test.isnull().sum()
train.describe()
test.describe()
train.groupby(['Survived']).mean()
train[['Survived','Age', 'Fare', 'SibSp']].groupby(['Survived']).mean()
train.groupby(['Pclass']).mean()
train[['Age','Fare', 'Sex', 'Survived']].groupby(['Sex', 'Survived']).mean()
train[['Pclass', 'Survived', 'Age']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train.pivot_table('Survived', index='Sex', columns='Pclass')
sns.barplot(x='Sex', y='Survived',data=train)
sns.barplot(x='Pclass', y='Survived', hue='Sex', palette='deep', data=train)
grid = sns.FacetGrid(train, size=5)

grid.map(sns.barplot, 'Pclass', 'Survived', palette='deep', order=[3,2,1], data=train)


sns.barplot(x='Sex', y='Survived', hue='Pclass', order=['male', 'female'], palette='deep', data=train)
grid = sns.FacetGrid(train, col='Pclass', col_wrap=4, size = 3)

grid.map(sns.barplot, 'Sex', 'Survived', order=['male','female'], palette='deep')
train.hist(column="Age",by="Pclass",bins=30)
#plt.hist(x='Survived', y='Sex', data=train)

grid = sns.FacetGrid(train, hue='Survived', size=3.5)

grid.map(plt.hist, 'Fare', alpha=.5)

grid.add_legend();
grid = sns.FacetGrid(train, col="Pclass", hue="Survived", size=3)

grid.map(plt.scatter, "Fare", "Age", alpha=.5)

plt.xlim(0,300)

grid.add_legend();
# create new data sets without n variables that seem irrelevant to the survivival rates

train_df = train.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)

test_df = test.drop(['Ticket', 'Cabin', 'Name'], axis=1)



#dataFull_df = train_df.append(test_df, ignore_index=True)



dataFull_df = [train_df, test_df]

train_df.shape, test_df.shape
train.tail()
train_df.isnull().sum()
# replaces age null values.



#train_df['Age'].fillna(train_df['Age'].dropna().mode()[0], inplace=True)

#train_df.isnull().sum()



train_df['Age'].fillna(train_df['Age'].dropna().median(), inplace=True)

train_df.isnull().sum()
train_df['Embarked'].fillna(train_df['Embarked'].dropna().mode()[0], inplace=True)

train_df.isnull().sum()
test_df.isnull().sum()
test_df['Age'].fillna(test_df['Age'].dropna().median(), inplace=True)

test_df.isnull().sum()
test_df['Fare'].fillna(test_df['Fare'].dropna().mode()[0], inplace=True)

test_df.isnull().sum()
test_df.head()
# Transform Sex into binary values 0 and 1

for dataset in dataFull_df:

    dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0}).astype(int)



train_df.head()



#sex = pd.Series( np.where( dataFull.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
# Create a new variable for every unique value of Embarked

for dataset in dataFull_df:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

#train_df.head()

train_df[['AgeBand', 'Survived']].groupby('AgeBand', as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in dataFull_df:

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
train_df.head()
train_df['Fareband'] = pd.qcut(train_df['Fare'], 4)

train_df[['Fareband', 'Survived']].groupby('Fareband', as_index=False).mean().sort_values(by='Fareband', ascending=True)
for dataset in dataFull_df:    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df.head()
train_df = train_df.drop('AgeBand', axis=1)

train_df = train_df.drop('Fareband', axis=1)



train_df.head(2)
train_df = train_df.drop('Embarked', axis=1)
train_df.head()


test_df = test_df.drop('PassengerId', axis=1)

test_df = test_df.drop('Embarked', axis=1)
test_df.head()
#setting up logistic regression

x_train = train_df.drop('Survived', axis=1)

y_train = train_df['Survived']

x_test = test_df



x_train.shape
x_test.shape
# logistic regression

log_clf = LogisticRegression() 

log_clf.fit(x_train, y_train)

y_pred_log_reg = log_clf.predict(x_test)

log_clf.score(x_train,y_train)
#support vector machine

svc = SVC()

svc.fit(x_train, y_train)

y_pred_svc = svc.predict(x_test)

svc.score(x_train, y_train) 
#random forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

y_pred_randomforest = random_forest.predict(x_test)

random_forest.score(x_train,y_train)
#dataFull_df = train_df.append(test_df, ignore_index=True)

#dataFull_df.head()