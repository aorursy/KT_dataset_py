import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
data = pd.read_csv('../input/titanic/train.csv')

data.head()
# check null portion

data.agg(lambda x: sum(x.isnull())/x.shape[0])
fig, ax = plt.subplots(1,2,figsize=(18,8))

data['Survived'].value_counts().plot.pie(autopct='%1.1f%%',ax = ax[0])

ax[0].set_ylabel('')

ax[0].set_title('Survived')



sns.countplot('Survived', data=data, ax=ax[1])

ax[1].set_title('Survived')

plt.show()
fig, ax = plt.subplots(1,2,figsize=(18,8))

data[['Sex', 'Survived']].groupby('Sex').mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')



sns.countplot('Sex', hue='Survived', data=data, ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
pd.crosstab(data.Pclass, data.Survived, margins=True)
fig, ax = plt.subplots(1,2,figsize=(18,8))

data['Pclass'].value_counts().plot.bar(ax=ax[0])



sns.countplot('Pclass', hue='Survived', data=data, ax=ax[1])

plt.show()
sns.factorplot('Pclass','Survived', hue='Sex', data=data)

plt.show()
data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')

data['Initial'].value_counts()
# you can use dictionary instead

data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
data[['Initial', 'Age']].groupby('Initial').mean()
data.loc[(data.Age.isnull())&(data.Initial=='Mr'), 'Age']=33

data.loc[(data.Age.isnull())&(data.Initial=='Mrs'), 'Age']=36

data.loc[(data.Age.isnull())&(data.Initial=='Master'), 'Age']=5

data.loc[(data.Age.isnull())&(data.Initial=='Miss'), 'Age']=22

data.loc[(data.Age.isnull())&(data.Initial=='Other'), 'Age']=46
data.Age.isnull().any()
sns.factorplot('Pclass', 'Survived', col='Initial', data=data)

plt.show()
sns.factorplot('Embarked', 'Survived', data=data)

plt.show()
sns.countplot('Embarked', hue='Pclass', data=data)

plt.show()
sns.factorplot('Pclass','Survived', hue='Sex', col='Embarked', data=data)

plt.show()
print(data['Embarked'].isnull().value_counts())

data['Embarked'].fillna('S',inplace=True)
fig, ax = plt.subplots(1,3,figsize=(20,8))

sns.distplot(data[data['Pclass']==1].Fare, ax=ax[0])

sns.distplot(data[data['Pclass']==2].Fare, ax=ax[1])

sns.distplot(data[data['Pclass']==3].Fare, ax=ax[2])

plt.show()
sns.heatmap(data.corr(), square=True, annot=True, cmap='RdBu', linewidth=0.2)

plt.show()
def category_age(x):

    if x<= 16:

        return 0

    elif x<=32:

        return 1

    elif x<=48:

        return 2

    elif x<=64:

        return 3

    else:

        return 4

    

data['Age_band'] = data['Age'].apply(category_age)

data['Age_band'].value_counts().to_frame()
data['Family_Size']=0

data['Family_Size'] = data['Parch']+data['SibSp']

data['Alone']=0

data.loc[data.Family_Size==0,'Alone']=1
sns.countplot('Alone', hue='Survived',data=data)

plt.show()
data['Fare_Range'] = pd.qcut(data['Fare'],4)

data[['Fare_Range','Survived']].groupby('Fare_Range').mean()
def category_fare(x):

    if x<=7.91:

        return 0

    elif x<=14.454:

        return 1

    elif x<=31:

        return 2

    else:

        return 3

    

data['Fare_cat'] = data['Fare'].apply(category_fare)
sns.factorplot('Fare_cat', 'Survived', hue='Sex', data=data)

plt.show()
#string to numeric values



data['Sex'].replace({'male':0, 'female':1}, inplace=True)

data['Embarked'].replace({'S':0, 'C':1, 'Q':2}, inplace=True)

data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'], axis=1, inplace=True)

plt.figure(figsize=(20,20))

sns.heatmap(data.corr(), annot=True, square=True, cmap='RdBu')

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier



from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix
train, valid = train_test_split(data, test_size=0.3, stratify=data['Survived'])

train.head(2)
train_X = train[train.columns[1:]]

train_Y = train[train.columns[:1]]

valid_X = valid[valid.columns[1:]]

valid_Y = valid[valid.columns[:1]]

X = data[data.columns[1:]]

Y = data[data.columns[:1]]
# SVM

model = svm.SVC(kernel='linear', C=0.1, gamma=0.1)

model.fit(train_X, train_Y)

prediction = model.predict(valid_X)

metrics.accuracy_score(prediction, valid_Y)
# Logistic regression

model = LogisticRegression()

model.fit(train_X, train_Y)

prediction = model.predict(valid_X)

metrics.accuracy_score(prediction, valid_Y)
# Decision Tree

model = DecisionTreeClassifier()

model.fit(train_X, train_Y)

prediction = model.predict(valid_X)

metrics.accuracy_score(prediction, valid_Y)
# KNN

model = KNeighborsClassifier(n_neighbors=5)

model.fit(train_X, train_Y)

prediction = model.predict(valid_X)

metrics.accuracy_score(prediction, valid_Y)
# Gaussian Naive Bayes

model = GaussianNB()

model.fit(train_X, train_Y)

prediction = model.predict(valid_X)

metrics.accuracy_score(prediction, valid_Y)
# Random FOrest

model = RandomForestClassifier(n_estimators=100)

model.fit(train_X, train_Y)

prediction = model.predict(valid_X)

metrics.accuracy_score(prediction, valid_Y)
# cross validation

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

kfold = KFold(n_splits=10)

means = []

stds = []

accuracies =[]

classifiers = ['Linear SVM', 'Logistic Regression', 'KNN', 'Decision Tree', 'Naive Bayes', 'Random Forest']

models = [svm.SVC(kernel='linear'), LogisticRegression(), KNeighborsClassifier(9),DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier(100)]



for model in models:

    cv_result = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')

    means.append(cv_result.mean())

    stds.append(cv_result.std())

    accuracies.append(cv_result)

    

cv_result_df = pd.DataFrame({'CV mean':means, 'Std':stds}, index=classifiers)
plt.figure(figsize=(16,6))

box = pd.DataFrame(accuracies, index=[classifiers])

box.T.boxplot()
fig, ax = plt.subplots(2,3, figsize=(12,10))



y_pred = cross_val_predict(svm.SVC(kernel='linear'), X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[0,0], annot=True, fmt='2.0f')

ax[0,0].set_title('Matrix for Linear_SVM')

y_pred = cross_val_predict(KNeighborsClassifier(9), X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[0,1], annot=True, fmt='2.0f')

ax[0,1].set_title('Matrix for KNN')

y_pred = cross_val_predict(RandomForestClassifier(), X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[0,2], annot=True, fmt='2.0f')

ax[0,2].set_title('Matrix for Random Forest')

y_pred = cross_val_predict(LogisticRegression(), X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[1,0], annot=True, fmt='2.0f')

ax[1,0].set_title('Matrix for Logistic Regression')

y_pred = cross_val_predict(DecisionTreeClassifier(), X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[1,1], annot=True, fmt='2.0f')

ax[1,1].set_title('Matrix for Decision Tree')

y_pred = cross_val_predict(GaussianNB(), X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[1,2], annot=True, fmt='2.0f')

ax[1,2].set_title('Matrix for Gaussian NB')
# hyper params tuning

from sklearn.model_selection import GridSearchCV

C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]

gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

kernel = ['linear']

hyper = {'kernel':kernel, 'C':C, 'gamma':gamma}

gd = GridSearchCV(estimator = svm.SVC(), param_grid=hyper, verbose=True)

gd.fit(X,Y)

print(gd.best_score_)

print(gd.best_estimator_)
n_estimators=range(100,1000,100)

hyper={'n_estimators':n_estimators}

gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)

gd.fit(X,Y)

print(gd.best_score_)

print(gd.best_estimator_)
# Ensemble - Vote

from sklearn.ensemble import VotingClassifier

ensemble_vote = VotingClassifier(estimators=[('KNN', KNeighborsClassifier(10)),

                                           ('RFor', RandomForestClassifier(500)),

                                           ('LR', LogisticRegression(C=0.05)),

                                           ('DT', DecisionTreeClassifier()),

                                           ('NB', GaussianNB()),

                                           ('svm',svm.SVC(kernel='linear', probability=True))]

                                , voting='soft').fit(train_X, train_Y)



print(ensemble_vote.score(valid_X, valid_Y))

cross = cross_val_score(ensemble_vote, X, Y, cv=10, scoring='accuracy')

print(cross.mean())

# Bagging - use similar clasifiers / lower variance

from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(base_estimator=KNeighborsClassifier(3), n_estimators=700)

model.fit(train_X, train_Y)

prediction = model.predict(valid_X)

print('The accuracy for bagged KNN is : ', metrics.accuracy_score(prediction, valid_Y))

result = cross_val_score(model, X, Y, cv=10, scoring='accuracy')

print('The cross validated score for bagged KNN is : ', result.mean())
# Boosting - sequential weak learners

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=200, learning_rate = 0.1)

result = cross_val_score(ada, X, Y, cv=10, scoring='accuracy')

result.mean()
from sklearn.ensemble import GradientBoostingClassifier

grad = GradientBoostingClassifier(n_estimators=500, learning_rate =0.1)

result = cross_val_score(grad, X, Y, cv=10, scoring='accuracy')

result.mean()
import xgboost as xg

xgboost = xg.XGBClassifier(n_estimators=900, learning_rate=0.1)

result = cross_val_score(xgboost, X, Y, cv=10, scoring='accuracy')

result.mean()
# Hyperparameter tuning with AdaBoost since we got the highest accuracy

n_estimators = list(range(100,1100,100))

learning_rate = learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]

hyper = {'n_estimators':n_estimators, 'learning_rate':learning_rate}

gd = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=hyper, verbose=True)

gd.fit(X,Y)

print(gd.best_score_)

print(gd.best_estimator_)
ada = AdaBoostClassifier(n_estimators = 200, learning_rate = 0.05)

result = cross_val_predict(ada, X, Y, cv=10)

sns.heatmap(confusion_matrix(Y, result), cmap='winter', annot=True, fmt='2.0f')

plt.show()
# Feature importance

fig, ax = plt.subplots(2,2, figsize=(15,12))

model = RandomForestClassifier(n_estimators=500)

model.fit(X,Y)

pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[0,0])

ax[0,0].set_title('Feature Importance in Random Forests')



model = AdaBoostClassifier(n_estimators=200, learning_rate=0.05)

model.fit(X,Y)

pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8, ax=ax[0,1])

ax[0,1].set_title('Feature Importance in AdaBoost')



model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')

ax[1,0].set_title('Feature Importance in Gradient Boosting')



model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)

model.fit(X,Y)

pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')

ax[1,1].set_title('Feature Importance in XgBoost')

plt.show()
test_data = pd.read_csv('../input/titanic/test.csv')

test_data['Initial'] = test_data.Name.str.extract('([A-Za-z]+)\.')

test_data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)



test_data['Initial'].replace(['Dona'], ['Mr'], inplace=True)



test_data.loc[(test_data.Age.isnull())&(test_data.Initial=='Mr'), 'Age']=33

test_data.loc[(test_data.Age.isnull())&(test_data.Initial=='Mrs'), 'Age']=36

test_data.loc[(test_data.Age.isnull())&(test_data.Initial=='Master'), 'Age']=5

test_data.loc[(test_data.Age.isnull())&(test_data.Initial=='Miss'), 'Age']=22

test_data.loc[(test_data.Age.isnull())&(test_data.Initial=='Other'), 'Age']=46

test_data['Embarked'].fillna('S',inplace=True)

test_data['Age_band'] = test_data['Age'].apply(category_age)

test_data['Family_Size']=0

test_data['Family_Size'] = test_data['Parch']+test_data['SibSp']

test_data['Alone']=0

test_data.loc[test_data.Family_Size==0,'Alone']=1

test_data['Fare_Range'] = pd.qcut(test_data['Fare'],4)

test_data['Fare_cat'] = test_data['Fare'].apply(category_fare)

test_data['Sex'].replace({'male':0, 'female':1}, inplace=True)

test_data['Embarked'].replace({'S':0, 'C':1, 'Q':2}, inplace=True)

test_data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

test_data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'], axis=1, inplace=True)



submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission.head(2)
test_data.head()
ada = AdaBoostClassifier(n_estimators = 200, learning_rate = 0.05)

ada.fit(X,Y)

prediction = ada.predict(test_data)

submission['Survived'] = prediction

submission.to_csv('./my_first_submission.csv', index=False)
import base64

import pandas as pd

from IPython.display import HTML



def create_download_link( df, title = "Download CSV file", filename = "./my_first_submission.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(submission)