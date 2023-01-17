# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import math



# visualization

import seaborn as sns

import matplotlib.pyplot as plt



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

import xgboost as xgb

from xgboost.sklearn import XGBClassifier # <3



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.info())
print(test.info())
train.head()
(test.head())
x = ((train.Age - train.Age.round())*10)

print(train.loc[x == 5])
x = ((test.Age - test.Age.round())*10)

print(test.loc[x > 1])
print(train.describe())
print(test.describe())
corr=train.corr()#["Survived"]

plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="white")

plt.title('Correlation between features');
sns.set(font_scale=1)

g = sns.factorplot(x="Sex", y="Survived", col="Pclass",

                    data=train, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate")

    .set_xticklabels(["Men", "Women"])

    .set_titles("{col_name} {col_var}")

    .set(ylim=(0, 1))

    .despine(left=True))  

plt.subplots_adjust(top=0.8)

g.fig.suptitle('How many Men and Women Survived by Passenger Class');
sns.set(font_scale=1)

g = sns.factorplot(x="Sex", y="Survived", col="Embarked",

                    data=train, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate")

    .set_xticklabels(["Men", "Women"])

    .set_titles("{col_name} {col_var}")

    .set(ylim=(0, 1))

    .despine(left=True))  

plt.subplots_adjust(top=0.8)

g.fig.suptitle('How many Men and Women Survived by Port of Embarkation');
figure = plt.figure(figsize=(15,8))

withAge = (train[train['Age'].notnull()])

plt.hist([withAge[withAge['Survived']==1]['Age'], withAge[withAge['Survived']==0]['Age']], stacked=True, color = ['g','r'],bins = 30,label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()
train.isnull().sum()
test.isnull().sum()
ageless = (train[train['Age'].isnull()])

withAge = (train[train['Age'].notnull()])

print ("Full training set")

print (train[["Sex", "Survived", "Pclass"]].groupby(['Sex', "Pclass"], as_index=False).mean())

print ("Training set with an age")

print (withAge[["Sex", "Survived", "Pclass"]].groupby(['Sex', "Pclass"], as_index=False).mean())

print ("Training set without an age")

print (ageless[["Sex", "Survived", "Pclass"]].groupby(['Sex', "Pclass"], as_index=False).mean())
cabinless = (train[train['Cabin'].isnull()])

withCabin = (train[train['Cabin'].notnull()])

print ("Full training set")

print (train[["Sex", "Survived", "Pclass"]].groupby(['Sex', "Pclass"], as_index=False).mean())

print ("Training set with a cabin")

print (withCabin[["Sex", "Survived", "Pclass"]].groupby(['Sex', "Pclass"], as_index=False).mean())

print ("Training set without a cabin")

print (cabinless[["Sex", "Survived", "Pclass"]].groupby(['Sex', "Pclass"], as_index=False).mean())
targets = train.Survived

train.drop('Survived', 1, inplace=True)
full_data = train.append(test)

full_data.reset_index(inplace=True)

full_data.drop('index', inplace=True, axis=1)
print(full_data.shape)

print(targets.shape)
full_data['EstimatedAge']=full_data.Age.isnull().astype(int)

full_data['EstimatedCabin']=full_data.Cabin.isnull().astype(int)
full_data.isnull().sum()
print(full_data[full_data['Fare'].isnull()])
grouped_test = full_data.iloc[891:].groupby(['Sex','Pclass'])

grouped_median_test = grouped_test.median()



print(grouped_median_test)
full_data.Fare.fillna(7.895, inplace=True)

full_data.Fare = full_data.Fare.astype(int)
print(train[train['Embarked'].isnull()])
full_data["Embarked"] = full_data["Embarked"].fillna('C')
embarked_dummies = pd.get_dummies(full_data['Embarked'],prefix='Embarked')

full_data = pd.concat([full_data,embarked_dummies],axis=1)

full_data = full_data.drop(["Embarked"], axis=1)
full_data.info()
full_data['Sex'] = full_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
full_data['Title'] = full_data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Jonkheer":   "Nobility",

                    "Don":        "Nobility",

                    "Sir" :       "Nobility",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "Countess":   "Nobility",

                    "Dona":       "Nobility",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Nobility"

                    }

full_data['Title'] = full_data.Title.map(Title_Dictionary)
full_data.drop('Name',axis=1,inplace=True)
titles_dummies = pd.get_dummies(full_data['Title'],prefix='Title')

full_data = pd.concat([full_data,titles_dummies],axis=1)
print(full_data.head())
age_train = full_data.head(891)[['Age','Sex', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Nobility', 'Title_Officer', 'Pclass']]

print(age_train.info())
known_train  = age_train.loc[ (age_train.Age.notnull()) ]# known Age values for training set

unknown_train = age_train.loc[ (age_train.Age.isnull()) ]# unknown values for training set
y_train = known_train.values[:, 0]

X_train = known_train.values[:, 1::]
rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

rtr.fit(X_train, y_train)

predictedAges = rtr.predict(unknown_train.values[:, 1::])
dup_train = full_data.head(891)

dup_train.loc[ (dup_train.Age.isnull()), 'Age' ] = predictedAges 

print(dup_train[dup_train['Age'].isnull()])
age_test = full_data.iloc[891:][['Age','Sex', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Nobility', 'Title_Officer', 'Pclass']]



# Split sets into known and unknown test

known_test  = age_test.loc[ (age_test.Age.notnull()) ]# known Age values for testing set

unknown_test = age_test.loc[ (age_test.Age.isnull()) ]# unknown values for testing set

   

# All age values are stored in target arrays

y_test = known_test.values[:, 0]

    

# All the other values are stored in the feature array

X_test = known_test.values[:, 1::]

    

# Create and fit a model

rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

rtr.fit(X_test, y_test)

    

# Use the fitted model to predict the missing values

predictedAgesTest = rtr.predict(unknown_test.values[:, 1::])



# Assign those predictions to the full data set

dup_test = full_data.iloc[891:]

dup_test.loc[ (dup_test.Age.isnull()), 'Age' ] = predictedAgesTest 

full_data = dup_train.append(dup_test)

full_data.reset_index(inplace=True)

full_data.drop('index', inplace=True, axis=1)

print(full_data.shape)
full_data.drop('Title',axis=1,inplace=True)
full_data.drop('Cabin',axis=1,inplace=True)
full_data.isnull().sum()
full_data['FamilySize'] = full_data['Parch'] + full_data['SibSp'] + 1
full_data['IsAlone'] = full_data['FamilySize'].map(lambda s: 1 if s == 1 else 0)
full_data = full_data.drop(["SibSp"], axis=1)

full_data = full_data.drop(["Parch"], axis=1)

full_data = full_data.drop(["Ticket"], axis=1)
X = full_data.head(891)

test = full_data.iloc[891:]

Y = targets
validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

seed = 7

scoring = 'accuracy'
# Spot Check Algorithms

models = []

models.append(('LogisticRegression', LogisticRegression()))

models.append(('KNeighborsClassifier', KNeighborsClassifier()))

models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))

models.append(('GaussianNB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('RandomForestClassifier', RandomForestClassifier()))

models.append(('XGBClassifier',  XGBClassifier()))

# evaluate each model in turn

results = []

mnames = []

for mname, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=seed)

	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

	results.append(cv_results)

	mnames.append(mname)

	msg = "%s: %f (%f)" % (mname, cv_results.mean(), cv_results.std())

	print(msg)
# Make predictions on validation dataset

lr = LogisticRegression()

lr.fit(X_train, Y_train)

predictions = lr.predict(X_validation)

print("Logistic Regression")

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))



knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

predictions = knn.predict(X_validation)

print("K Nearest Neighbours")

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))



dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)

predictions = dt.predict(X_validation)

print("Decision Tree Classifier")

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))



nb = GaussianNB()

nb.fit(X_train, Y_train)

predictions = nb.predict(X_validation)

print("Gaussain NB")

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))



svm = SVC()

svm.fit(X_train, Y_train)

predictions = svm.predict(X_validation)

print("Support Vector Machine")

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))



rf = RandomForestClassifier()

rf.fit(X_train, Y_train)

predictions = rf.predict(X_validation)

print("Random Forest")

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))



XGB = XGBClassifier()

XGB.fit(X_train, Y_train)

predictions = XGB.predict(X_validation)

print("XGBoost")

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))



X_train = X.drop(["PassengerId"], axis=1)

Y_train = Y

X_test  = test.drop(["PassengerId"], axis=1)

print(X_train.shape, Y_train.shape, X_test.shape)



# LR = LogisticRegression()

# LR.fit(X_train, Y_train)

# Y_pred = LR.predict(X_test)

# LR.score(X_train, Y_train)

# acc_LR = round(LR.score(X_train, Y_train) * 100, 2)

# print(acc_LR)



from xgboost import XGBRegressor



my_model = XGBClassifier(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=5)

my_model.fit(X_train, Y_train)



Y_pred = my_model.predict(X_test)

# We will look at the predicted survival to ensure we have something sensible.

print(Y_pred)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

# print(submission)

submission.to_csv('submission.csv', index=False)