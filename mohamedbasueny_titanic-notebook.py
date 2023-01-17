#data proccesing and cleaning 

import numpy as np

import pandas as pd 

#data visulization  

%matplotlib inline 

import matplotlib.pyplot as plt 

import matplotlib as mpl

import seaborn as sns 



#ml model 

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB





train_df = pd.read_csv('./titanic/train.csv') #just replace dir with (../input/'name of the df')

test_df = pd.read_csv('./titanic/test.csv')
train_df.info()
train_df.head(10)
train_df.describe(include='all').transpose()
train_df.isnull().sum()
train_df.Age
total = train_df.isnull().sum().sort_values(ascending=False)

per_1 = train_df.isnull().sum() / train_df.isnull().count() *100

per_2 = (round(per_1,1)).sort_values(ascending=False)

missing = pd.concat([total,per_2],axis=1,keys=['total','%'])

missing.head()
all_count = train_df.isnull().count() #isnull to also conclude Nan values

train_df.columns.values

men = train_df[train_df['Sex']=='male']

men
women = train_df[train_df['Sex'] == 'female'].Age.dropna() #Return a new Series with missing values removed

women # use ctrl+i for docs 
survived = 'Survived' #have value 1

wasted = 'Wasted' # have value 0

fig,axes =plt.subplots(nrows=1,ncols=2,figsize=(10,5))

women = train_df[train_df['Sex']=='female']

men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(),bins=15,label=survived,kde=False,ax=axes[0])

ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(),bins=15,label=wasted,kde=False,ax=axes[0])

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(),bins=18,label=survived,kde=False,ax=axes[1])

ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(),bins=18,label=wasted,kde=False,ax=axes[1])

ax.legend()

ax.set_title('Male')
sns.barplot(x='Pclass',y='Survived',data=train_df)
grid = sns.FacetGrid(train_df,row='Pclass',col='Survived',height=2.5,aspect=1.5)

grid.map(plt.hist,'Age',alpha=.7,bins=20)

grid.add_legend()
data = [train_df,test_df]

for d in data :

    d['relatives'] = d['SibSp'] + d['Parch']

    d.loc[d['relatives']>0, 'not_alone'] = 0

    d.loc[d['relatives']==0,'not_alone'] = 1

    d['not_alone'] = d['not_alone'].astype(int)



    
train_df.info
axes = sns.catplot(x='relatives',y='Survived',data=train_df,aspect=2.5,kind='point')

#drop passengerId

train_df = train_df.drop(['PassengerId'],axis=1)
train_df.Cabin
train_df['Cabin'].fillna('U0')
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train_df, test_df]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int) 
data=[train_df,test_df]

for dataset in data :

    mean = dataset['Age'].mean()

    std = dataset["Age"].std()

    is_null = dataset['Age'].isnull().sum()

# compute random numbers between the mean, std and is_null

    range_num = np.random.randint(mean-std,mean+std,size=is_null)

# fill NaN values in Age column with random values generated

    age = dataset["Age"].copy()

    age[np.isnan(age)]=range_num

    dataset['Age']=age

    dataset["Age"] = dataset["Age"].astype(int)

    
train_df.Age.isnull().sum()
train_df.Embarked.describe()
train_df.Embarked.isnull().sum()
#let's replace Nan

common_value = 'S'

data=[train_df,test_df]

for dataset in data :

    dataset['Embarked'].fillna(common_value)
train_df.info()
#converting fare type into int

data = [train_df, test_df]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
train_df[train_df['Fare']==0].count()
train_df.Name
#Name we need it as aGender 

data = [train_df, test_df]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)
#drop Name

train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)
train_df.Title
data = [train_df,test_df]

Genders = {'male':1,'female':0}

for dataset in data :

    dataset['Sex'] = dataset['Sex'].map(Genders)
train_df.Sex
train_df.Ticket.describe()
#let's drop ticket columns many unique values

train_df.drop(['Ticket'],axis=1)



test_df.drop(['Ticket'],axis=1)
train_df.drop(['Cabin'],axis=1)
train_df = train_df.drop(['Cabin'],axis=1)

test_df = test_df.drop(['Cabin'],axis=1)
train_df = train_df.drop(['Ticket'],axis=1)

test_df = test_df.drop(['Ticket'], axis=1)
train_df
pla = pd.read_csv('./titanic/train.csv')

pla2 = pd.read_csv('./titanic/test.csv')

train_df['Embarked'] = pla['Embarked']

test_df['Embarked'] = pla2['Embarked']

train_df
train_df.Embarked=train_df.Embarked.fillna('S')
ports = {"S": 0, "C": 1, "Q": 2}

data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)

    
train_df.isnull().sum()
train_df.Embarked
train_df.Embarked.isnull().sum()
train_df.Embarked[:20]
train_df
pla['Age'].describe()
data = [train_df, test_df]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 66
# let's see how it's distributed

train_df['Age'].value_counts()
train_df
train_df['Fare'].describe()
data = [train_df, test_df]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)
train_df
#how test.csv are doin here

test_df
test_df
train_df
x_train = train_df.drop('Survived',axis=1)

y_train = train_df['Survived']

x_test = test_df.drop('PassengerId',axis=1).copy()
#sgd algorithm 

sgd = linear_model.SGDClassifier(max_iter=5,tol=None)

sgd.fit(x_train,y_train)

predictions = sgd.predict(x_test)

acc_sgd = round(sgd.score(x_train,y_train)*100,2) #the score between x and y ,it's an apgrade of LR

print(acc_sgd, "%")

#randomforest

randomForest = RandomForestClassifier(n_estimators=100,oob_score=True)

randomForest.fit(x_train,y_train)

ran_predict = randomForest.predict(x_test)

acc_ran = round(randomForest.score(x_train,y_train)*100,2)

print(acc_ran,'%')
#LogisticRegression 

log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)

log_predict = log_reg.predict(x_test)

acc_log = round(log_reg.score(x_train,y_train)*100,2)

print(acc_log,'%')
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(x_train, y_train)



Y_pred = gaussian.predict(x_test)



acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

print(acc_gaussian, "%")
# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)



Y_pred = linear_svc.predict(x_test)



acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)

print(acc_linear_svc, "%")
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train, y_train)



Y_pred = decision_tree.predict(x_test)



acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

print(acc_decision_tree ,"%")
# KNN

knn = KNeighborsClassifier(n_neighbors = 2)

knn.fit(x_train, y_train)



Y_pred = knn.predict(x_test)



acc_knn = round(knn.score(x_train, y_train) * 100, 2)

print(acc_knn, "%")
Models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes',  

              'Stochastic Gradient Decent', 

              'Decision Tree'],

    'Score': [acc_linear_svc, acc_knn, acc_log, 

              acc_ran, acc_gaussian, 

              acc_sgd, acc_decision_tree]})

models_df = Models.sort_values(by='Score',ascending=False)

models_df = models_df.set_index('Score')

models_df
from sklearn.model_selection import cross_val_score

forest = RandomForestClassifier(n_estimators=100)

score = cross_val_score(forest,x_train,y_train,cv=10,scoring='accuracy')

score
print("Scores:", score)

print("Mean:", score.mean())

print("Standard Deviation:", score.std())
importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(randomForest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances
importances.plot.bar()
train_df  = train_df.drop("not_alone", axis=1)

test_df  = test_df.drop("not_alone", axis=1)



train_df  = train_df.drop("Parch", axis=1)

test_df  = test_df.drop("Parch", axis=1)
random_forest = RandomForestClassifier()
print("oob score:", round(randomForest.oob_score_, 4)*100, "%")
#random forest 

random_forest = RandomForestClassifier(

    n_estimators=100,

    criterion='gini',

    oob_score=True,

    min_samples_split=10,

    min_samples_leaf=1,

    max_features='auto',

    random_state=1,

    n_jobs=-1 )

random_forest.fit(x_train,y_train)

y_predict = random_forest.predict(x_test)

print('accuracy: ',round(random_forest.score(x_train,y_train)*100,2),'%')

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": y_predict

    })

submission.to_csv('submission.csv', index=False)
pd.read_csv('./submission.csv')