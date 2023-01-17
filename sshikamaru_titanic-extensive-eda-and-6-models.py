#basic imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.walk('/kaggle/input'))
#Import Train and Test data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()
test_data.head()

#Coorlation of people who survived given that they are a female

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
def bar_chart(feature):

    survived = train_data[train_data['Survived']==1][feature].value_counts()

    dead = train_data[train_data['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))


train_test_data = [train_data, test_data] # combining train and test dataset



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)





title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)

print(train_data['Title'])

train_data.head()

bar_chart('Title')
sex_mapping = {"male": 0, "female":1}

for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

print(train_data['Sex'])
bar_chart('Sex')
train_data.head(10)

train_data["Age"].fillna(train_data.groupby("Title")["Age"].transform("median"), inplace=True)

test_data["Age"].fillna(test_data.groupby("Title")["Age"].transform("median"), inplace=True)

train_data.head(10)


facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train_data['Age'].max()))

facet.add_legend()

 

plt.show()
for dataset in train_test_data:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,

    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,

    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,

    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
train_data.head(100)
bar_chart('Age')
Pclass1 = train_data[train_data['Pclass'] == 1]['Embarked'].value_counts()

Pclass2 = train_data[train_data['Pclass'] == 2]['Embarked'].value_counts()

Pclass3 = train_data[train_data['Pclass'] == 3]['Embarked'].value_counts()



df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))
for data in train_test_data:

    data['Embarked'] = data['Embarked'].fillna("S")

    

embarked_mapping = {"S": 0, "C": 1, "Q": 2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

train_data.head(10)
train_data["Fare"].fillna(train_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test_data["Fare"].fillna(test_data.groupby("Pclass")["Fare"].transform("median"), inplace=True)

train_data.head(20)
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train_data['Fare'].max()))

facet.add_legend()

 

plt.show()


facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train_data['Fare'].max()))

facet.add_legend()

plt.xlim(0, 30)
for dataset in train_test_data:

    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,

    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,

    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,

    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
train_data.Cabin.value_counts()

test_text = 'abc'

print(test_text[:1])

for data in train_test_data:

    data["Cabin"] = data["Cabin"].str[:1]

train_data.head()
Pclass1 = train_data[train_data['Pclass'] == 1]['Cabin'].value_counts()

Pclass2 = train_data[train_data['Pclass'] == 2]['Cabin'].value_counts()

Pclass3 = train_data[train_data['Pclass'] == 3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1,Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5))

train_data.head()
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

    
#fill in missing values

train_data["Cabin"].fillna(train_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

test_data["Cabin"].fillna(test_data.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train_data["FamilySize"] = train_data['SibSp'] + train_data['Parch'] + 1

test_data["FamilySize"] = test_data['SibSp'] + test_data['Parch'] + 1


facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'FamilySize',shade= True)

facet.set(xlim=(0, train_data['FamilySize'].max()))

facet.add_legend()

plt.xlim(0)
#adding it to the dataset

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for dataset in train_test_data:

    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
train_data.head()
features_drop = ['Ticket', 'SibSp', 'Parch']

train_data = train_data.drop(features_drop, axis =1)

test_data = test_data.drop(features_drop, axis=1)

train_data = train_data.drop(['PassengerId'], axis=1)
train_data.head()
test_data.head()
bar_chart('Pclass')

train_data.shape

test_data.shape
#imports

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn import model_selection

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import accuracy_score



import numpy as np

train_data.info
train_data.head()
target = train_data['Survived']

id = test_data['PassengerId']



train_data.drop('Survived', axis=1, inplace = True)

train_data.drop('Name', axis=1, inplace = True)



test_data.drop('Name', axis=1, inplace = True)

test_data.head()

train_data.head()



train = train_data

test = test_data
#X_train, X_test, y_train, y_train = train_test_split(train_data, survived, random_state=0, test_size=0.2, shuffle=False)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)





train_data.shape, target.shape

train_data.head()

test_data.head()
test.head()
#coltest = train.drop('Cabin', axis=1)

#coltest = coltest.drop('Embarked', axis=1)

#coltest = coltest.drop('Title', axis=1)

#coltest = coltest.drop('Age', axis=1)





clf = KNeighborsClassifier(n_neighbors = 50)

scoring = 'accuracy'

score = cross_val_score(clf, train, target, cv = k_fold, n_jobs = 1, scoring = scoring)

print(score)

print(np.mean(score))



clf = LogisticRegression()

scoring = 'accuracy'

score = cross_val_score(clf, train, target, cv = k_fold, n_jobs = 1, scoring = scoring)

print(score)

print(np.mean(score))
from sklearn.linear_model import LogisticRegression

clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, train, target, cv = k_fold, n_jobs = 1, scoring = scoring)

print(score)

print(np.mean(score))
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)

scoring = 'accuracy'

score = cross_val_score(clf, train, target, cv = k_fold, n_jobs = 1, scoring = scoring)

print(score)

print(np.mean(score))
clf = GaussianNB()

scoring = 'accuracy'

score = cross_val_score(clf, train, target, cv = k_fold, n_jobs = 1, scoring = scoring)

print(score)

print(np.mean(score))
train.head()
test.head()
clf = SVC()

scoring = 'accuracy'

score = cross_val_score(clf, train, target, cv = k_fold, n_jobs = 1, scoring = scoring)

print(score)

clf.fit(train, target)

print(np.mean(score))



y_pred = clf.predict(test)



pred=pd.DataFrame(y_pred)

print(pred)
'''xgb = xgb.XGBClassifier(random_state=0)

xgb.fit(train, target)

preds = xgb.predict(test)

print('XGBoost: ', accuracy_score(y_val, preds))



def create_model(trial):

    max_depth = trial.suggest_int("max_depth", 2, 30)

    n_estimators = trial.suggest_int("n_estimators", 1, 500)

    learning_rate = trial.suggest_uniform('learning_rate', 0.0000001, 1)

    gamma = trial.suggest_uniform('gamma', 0.0000001, 1)

    model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, gamma=gamma, random_state=0)

    return model



def objective(trial):

    model = create_model(trial)

    model.fit(X, y)

    preds = model.predict(X_val)

    score = accuracy_score(y_val, preds)

    return score



study = optuna.create_study(direction="maximize", sampler=sampler)

study.optimize(objective, n_trials=50)



xgb_params = study.best_params

xgb_params['random_state'] = 0

'''
'''xgb = XGBClassifier(**xgb_params)

xgb.fit(X, y)

#preds = xgb.predict(X_val)

#print('Optimied XGBoost: ', accuracy_score(y_val, preds))'''





pred=pd.DataFrame(y_pred)

sub = pd.concat([test_data['PassengerId'],pred], axis=1)

sub.columns=['PassengerId','Survived']

sub.to_csv('submission66.csv', index = False)

print(sub.to_csv)


#submission = pd.read_csv('submission1.csv')

#submission.head()
clf = DecisionTreeClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, train, target, cv = k_fold, n_jobs = 1, scoring = scoring)

print(score)

print(np.mean(score))