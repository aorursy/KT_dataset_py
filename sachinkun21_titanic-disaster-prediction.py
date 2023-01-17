# linear algebra

import numpy as np

# data processing

import pandas as pd



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Algortithmic packages

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier



import os

import warnings

warnings.filterwarnings('ignore')
# from google.colab import drive

# drive.mount('/content/drive')
print(os.listdir("../input/titanic"))
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')



#printing first 5 rows of data

train_df.head()
train_df.info()
# let's display the columns names

train_df.columns
train_df.describe()
test_df.iloc[152]
train_df.head()
# count of null values in each column

count  = train_df.isnull().sum().sort_values(ascending = False)



# percentage of Null Values in each column

percent = train_df.isnull().sum()/train_df.count()*100



# rounding and arranging the percentage

percent = round(percent,2).sort_values(ascending = False)



# concatenating count and percentage into one

missing_data = pd.concat([count,percent], axis = 1)



# printing top 5 rows

missing_data.head()
survived = 'survided'

not_survived = 'not_survived'

fig, axes = plt.subplots(nrows = 1 , ncols = 2 , figsize = (18,8))

women = train_df[train_df['Sex'] == 'female']

men = train_df[train_df['Sex'] == 'male']



ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=8, label = survived , ax=axes[0], kde = False)

ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins = 40, label=not_survived, ax = axes[0], kde =False)

ax.set_title('Female')



ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=8, label = survived , ax=axes[1], kde = False)

ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins = 40, label=not_survived, ax = axes[1], kde =False)

_ = ax.set_title('Male')

ax.legend()

plt.show()
facetgrid = sns.FacetGrid(train_df , row = 'Embarked', height = 4.5 , aspect =1.8)

facetgrid.map(sns.pointplot, 'Pclass','Survived', 'Sex', order=None, hue_order=None)

facetgrid.add_legend()

plt.show()
sns.barplot(x = 'Pclass', y = 'Survived' , data = train_df)

plt.show()
grid = sns.FacetGrid(train_df, row = 'Pclass', col='Survived', hue_order=None, height = 3, aspect=2)

grid.map(plt.hist, 'Age', alpha=0.7, bins = 20)

plt.show()

data = [train_df , test_df]

for dataset in data:

  dataset['Relatives']=dataset['Parch']+dataset['SibSp']

  dataset.loc[dataset['Relatives']>0,'Alone']=0

  dataset.loc[dataset['Relatives']==0,'Alone']=1

  dataset['Alone']=dataset['Alone'].astype(int)
train_df.head()
train_df.Alone.value_counts()
plt.figure(figsize=(16,7))

sns.pointplot(x='Relatives', y = 'Survived', data= train_df )

plt.show()
train_df = train_df.drop(['PassengerId'], axis = 1)

train_df.head()
import re



deck =  {'A':1  , 'B': 2 , 'C': 3, 'D':4 ,'E' : 5 , 'F':6 , 'G':7 , 'U':8}

data = [train_df , test_df]

for dataset in data:

  dataset['Cabin']=dataset['Cabin'].fillna('U0')

  dataset['Deck']=dataset['Cabin'].map(lambda x : re.compile("([a-zA-Z]+)").search(x).group())

  dataset['Deck']=dataset['Deck'].map(deck)

  dataset['Deck']=dataset['Deck'].fillna(0)

  dataset['Deck']=dataset['Deck'].astype(int)
train_df.Deck.value_counts()
# Same code as above, Regular Expression Simplified



import re



deck =  {'A':1  , 'B': 2 , 'C': 3, 'D':4 ,'E' : 5 , 'F':6 , 'G':7 , 'U':8}

data = [train_df , test_df]



for dataset in data:

  dataset['Cabin']=dataset['Cabin'].fillna('U0')

  dataset['Deck']=dataset['Cabin'].map(lambda x : x[0])

  dataset['Deck']=dataset['Deck'].map(deck)

  dataset['Deck']=dataset['Deck'].fillna(0)

  dataset['Deck']=dataset['Deck'].astype(int)



  

  

train_df.Deck.value_counts()
data = [train_df,test_df]

for dataset in data:

  dataset=dataset.drop(['Cabin'], axis = 1)
train_df=train_df.drop('Cabin', axis = 1)

test_df=test_df.drop('Cabin', axis = 1)
data = [train_df , test_df]

mean = train_df['Age'].mean()

std  = test_df['Age'].std()





for dataset in data:

  count_of_null = dataset['Age'].isnull().sum()

  

  rand_age = np.random.randint(mean-std,mean+std, size = count_of_null)

  

  age_slice = dataset['Age'].copy()

  age_slice[np.isnan(age_slice)]= rand_age

  

  dataset['Age']=age_slice

  dataset['Age']=dataset['Age'].astype(int)
train_df['Embarked'].describe()
train_df['Embarked'] = train_df['Embarked'].fillna('S')

test_df['Embarked'] = test_df['Embarked'].fillna('S')
train_df.info()
test_df.info()
train_df.Name.head()
data = [train_df,test_df]



for dataset in data:

  dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand = False)
train_df.Title.value_counts()
data = [train_df,test_df]



for dataset in data:

  dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don','Dr',\

                                              'Major','Rev','Sir','Jonkheer','Dona'], 'Rare')

  dataset['Title'] = dataset['Title'].replace('Mlle' , 'Miss')

  dataset['Title'] = dataset['Title'].replace('Ms' , 'Miss')

  dataset['Title'] = dataset['Title'].replace('Mme' , 'Mrs')

  
titles={'Mr':1,'Miss':2,'Mrs':3, 'Master':4, 'Rare':5}



for dataset in data:

  dataset['Title']=dataset['Title'].map(titles)
print(train_df.Title.isna().sum())

print(test_df.Title.isna().sum())
train_df = train_df.drop(['Name'], axis = 1)

test_df = test_df.drop(['Name'], axis = 1)
train_df.Sex.value_counts()
gender = {'male':0 , 'female':1}

data = [train_df,test_df]



for dataset in data:

  dataset['Sex']=dataset['Sex'].map(gender)
train_df.Ticket.head()
data = [train_df,test_df]

for dataset in data:

  dataset=dataset.drop('Ticket', axis = 1)
train_df=train_df.drop('Ticket', axis = 1)

test_df=test_df.drop('Ticket', axis = 1)
ports = {'S':0,'C':1, 'Q':2}

data = [train_df,test_df]



for dataset in data:

  dataset['Embarked']=dataset['Embarked'].map(ports)
data = [train_df,test_df]

for dataset in data:

  dataset['Age']=dataset['Age'].astype(int)

  dataset.loc[dataset['Age']<=11, 'Age']=0

  dataset.loc[(dataset['Age']>11) & (dataset['Age']<=18), 'Age']=1

  dataset.loc[(dataset['Age']>18) & (dataset['Age']<=22), 'Age']=2

  dataset.loc[(dataset['Age']>22) & (dataset['Age']<=27), 'Age']=3

  dataset.loc[(dataset['Age']>27) & (dataset['Age']<=33), 'Age']=4

  dataset.loc[(dataset['Age']>33) & (dataset['Age']<=40), 'Age']=5

  dataset.loc[(dataset['Age']>40) & (dataset['Age']<=66), 'Age']=6

  dataset.loc[(dataset['Age']>66), 'Age']=6
train_df.Age.value_counts()
train_df.head()
data = [train_df,test_df]



for dataset in data:

  dataset.loc[dataset['Fare']<=7.91, 'Fare']=0

  dataset.loc[(dataset['Fare']>7.91) & (dataset['Fare']<=14.454), 'Fare']=1

  dataset.loc[(dataset['Fare']>14.454) & (dataset['Fare']<=31), 'Fare']=2

  dataset.loc[(dataset['Fare']>31) & (dataset['Fare']<=99), 'Fare']=3

  dataset.loc[(dataset['Fare']>99) & (dataset['Fare']<=250), 'Fare']=4

  dataset.loc[(dataset['Fare']>250) , 'Fare']=5

  dataset['Fare']=dataset['Fare'].fillna(0)

  dataset['Fare']=dataset['Fare'].astype(int)



train_df.Fare.value_counts()
# test_df[test_df.Fare.isna()==True]
data = [train_df, test_df]





for dataset in data:

  dataset['age_class'] = dataset['Age']*dataset['Pclass']

  

  

train_df.head()
# train_df.head()
X_train = train_df.drop('Survived', axis = 1)

y_train = train_df['Survived']



X_test = test_df.drop('PassengerId' ,  axis = 1)
# creating model object

sgd = linear_model.SGDClassifier(max_iter = 5, tol = None)



# Fitting model on Data

sgd.fit(X_train,y_train)



#using model to predict

y_pred = sgd.predict(X_test)



# Storing prediction accuracy

acc_sgd = round(sgd.score(X_train,y_train)*100,2)

print(acc_sgd)
logreg = LogisticRegression()

logreg.fit(X_train,y_train)





acc_log=round(logreg.score(X_train,y_train)*100,2)

print(acc_log)
dict_model = {'sgd':linear_model.SGDClassifier(max_iter = 5, tol = None), 

             'log_reg':LogisticRegression(),

             'decision_tree':DecisionTreeClassifier(),

             'random_forest':RandomForestClassifier(n_estimators = 100),

             'knn_classifier': KNeighborsClassifier(n_neighbors= 3),

             'gaussian':GaussianNB(),

             'perceptron':Perceptron(max_iter=5),

             'linear_svc':LinearSVC()

             }
dict_accuracies={}



for name,classifier in dict_model.items():

  dict_model[name].fit(X_train,y_train)

  score = dict_model[name].score(X_train,y_train)

  dict_accuracies[name]=round(score*100,2)

  

result_df=pd.DataFrame.from_dict(dict_accuracies, orient = 'index',columns = ['Score'])

result_df= result_df.sort_values(by = 'Score', ascending = False)

result_df