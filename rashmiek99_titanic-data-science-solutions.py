import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier



from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
titanic_data = pd.concat([train,test],keys=['train','test'])

train_labels = train['Survived']

titanic_data.head()
train.info()

print('_'*40)

test.info()
train.describe()
train.describe(include=['O'])
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.countplot(x='Pclass',hue='Survived',data = train)

plt.title("Pclass Vs Survived")

plt.ylabel("")

plt.show()
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.countplot(x='Sex',hue='Survived',data = train)

plt.title("Sex Vs Survived")

plt.ylabel("")

plt.show()
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.countplot(x='SibSp',hue='Survived',data = train)

plt.title("SibSp Vs Survived")

plt.ylabel("")

plt.show()
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.countplot(x='Parch',hue='Survived',data = train)

plt.title("Parch Vs Survived")

plt.ylabel("")

plt.show()
g = sns.FacetGrid(train,col='Survived')

g.map(plt.hist,'Age')

plt.show()
train_df = train.drop(['PassengerId'], axis=1)



test_df = test.copy()

combine = [train_df, test_df]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Unknown')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Unknown": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)
train_df = train_df.drop(['Ticket', 'Cabin','Name'], axis=1)



test_df = test_df.drop(['Ticket', 'Cabin','Name'], axis=1)

combine = [train_df, test_df]
mean_age = train_df['Age'].mean()

std_age = test_df['Age'].std()

    

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

       

    null_age_count = dataset['Age'].isnull().sum()

    random_age = np.random.randint((mean_age - std_age),(mean_age + std_age),size=null_age_count)

    

    age_guess = dataset['Age'].copy()

    age_guess[np.isnan(age_guess)] = random_age

    

    dataset['Age'] = age_guess

    dataset['Age'] = dataset['Age'].astype(int)

    

combine = [train_df, test_df]
for dataset in combine:   

    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



for dataset in combine:

    size = []

    for col in dataset.FamilySize:

        if col > 1:

            size.append(0)

        else:

            size.append(1)

            

    dataset['Isalone'] = size

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]

sns.countplot(x='Isalone',hue='Survived',data=train_df);
for dataset in combine:

    dataset['Embarked']  = dataset['Embarked'].fillna('S')

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)    

    

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] >7.91) & (dataset['Fare']<=14.454),'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3



combine = [train_df, test_df]    

train_df.head()
X = train_df.drop("Survived", axis=1)

Y = train_df["Survived"]



X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)



Final_test  = test_df.drop("PassengerId", axis=1).values

X_train.shape,y_train.shape,X_test.shape,y_test.shape
X = train_df.drop(['Survived','Isalone'], axis=1).values

Y = train_df['Survived'].values



X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)



Final_test  = test_df.drop(['Isalone','PassengerId'], axis=1).values
lr =  LogisticRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)



lr_acc = round(accuracy_score(lr_pred, y_test), 3)

print('Logistic Regression accuracy: ',lr_acc)
dc =  DecisionTreeClassifier()

dc.fit(X_train, y_train)

dc_pred = dc.predict(X_test)



dc_acc = round(accuracy_score(dc_pred, y_test), 3)

print('Decision Tree Accuracy: ',dc_acc)
rfc = RandomForestClassifier(n_estimators=400,oob_score=True,criterion='entropy',random_state=1)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)



rfc_acc = round(accuracy_score(rfc_pred, y_test), 3)

print('Random Forest Classifier Accuracy: ',rfc_acc)
svc = SVC()

svc.fit(X_train, y_train)

svc_pred = svc.predict(X_test)



svc_acc = round(accuracy_score(svc_pred, y_test), 3)

print('Support Vector Machine Accuracy: ',svc_acc)
from sklearn.ensemble import BaggingClassifier

Bagg_estimators = [10,25,50,75,100,150,250];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.33, random_state=15)



parameters = {'n_estimators':Bagg_estimators }

gridBG = GridSearchCV(BaggingClassifier(base_estimator= None, ## If None, then the base estimator is a decision tree.

                                      bootstrap_features=False),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

gridBG.fit(X_train,y_train)



print ('Bagging classifier best score: ', gridBG.best_score_)

print (gridBG.best_params_)

print (gridBG.best_estimator_)



bagging = gridBG.best_estimator_



bagging_pred = bagging.predict(X_test)



bagging_acc = round(accuracy_score(bagging_pred, y_test), 3)

print('AdaBoost Accuracy: ',bagging_acc)


cv = StratifiedShuffleSplit(n_splits=10,test_size=0.3,random_state=15)



n_estimators = [100,140,145,150,160, 170,175,180,185];



learning_rate = [0.1,1,0.01,0.5]



parameters = {'n_estimators':n_estimators,'learning_rate':learning_rate}



gridAda = GridSearchCV(AdaBoostClassifier(base_estimator=None),param_grid=parameters,cv=cv,n_jobs=-1)



gridAda.fit(X_train,y_train)



print("Ada Boost Score:",gridAda.best_score_)

print(gridAda.best_params_)



adaBoost = gridAda.best_estimator_



#print('Ada Boost estimator score:',adaBoost.score(X_test,y_test))



adaBoost_pred = adaBoost.predict(X_test)



adaBoost_acc = round(accuracy_score(adaBoost_pred, y_test), 3)

print('AdaBoost Accuracy: ',adaBoost_acc)
gbc = GradientBoostingClassifier()

gbc.fit(X_train,y_train)



gbc_pred = gbc.predict(X_test)





gradient_acc = round(accuracy_score(gbc_pred, y_test), 3)

print('Gradient Boosting accuracy: ',gradient_acc)
clf = [lr, dc, rfc, svc,bagging,adaBoost,gbc]



for clf, label in zip([lr, dc, rfc, svc,bagging,adaBoost,gbc], ['Logistic Regression','Decision Tree','Random Forest', 'SVC','Bagging','AdaBoost','Gradient Boost']):

    scores = cross_val_score(clf, X_test, y_test, cv=10, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
eclf = VotingClassifier(estimators=[('Logistic Regression',lr),('Decision Tree',dc),('Random Forests', rfc),  ('SVC', svc), ('Bagging', bagging), ('AdaBoost', adaBoost), ('Gradient Boosting', gbc)], voting='hard')

eclf.fit(X_train,y_train)

vote_pred = eclf.predict(X_test)

voting_acc = round(accuracy_score(vote_pred,y_test),4)



print('Voting accuracy of the combined classifiers: ',voting_acc)
Y_Pred = eclf.predict(Final_test)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_Pred

    })

submission.to_csv('my_titanic_submission_voting.csv', index=False)