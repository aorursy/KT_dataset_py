#data analysis libraries 

import numpy as np

import pandas as pd



#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#ignore warnings

import warnings

warnings.filterwarnings('ignore')
#import train and test CSV files

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#take a look at the training data

train.describe(include="all")
#get a list of the features within the dataset

print(train.columns)
#see a sample of the dataset to get an idea of the variables

train.sample(5)
#see a summary of the training dataset

train.describe(include = "all")
#check for any other unusable values

print(pd.isnull(train).sum())
#draw a bar plot of survival by sex

sns.barplot(x="Sex", y="Survived", data=train)



#print percentages of females vs. males that survive

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
#draw a bar plot of survival by Pclass

sns.barplot(x="Pclass", y="Survived", data=train)



#print percentage of people by Pclass that survived

print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
#draw a bar plot for SibSp vs. survival

sns.barplot(x="SibSp", y="Survived", data=train)



#I won't be printing individual percent values for all of these.

print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)
#draw a bar plot for Parch vs. survival

sns.barplot(x="Parch", y="Survived", data=train)

plt.show()
#sort the ages into logical categories

train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

test["CabinBool"] = (test["Cabin"].notnull().astype('int'))



#calculate percentages of CabinBool vs. survived

print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)

#draw a bar plot of CabinBool vs. survival

sns.barplot(x="CabinBool", y="Survived", data=train)

plt.show()
test.describe(include="all")
#we'll start off by dropping the Cabin feature since not a lot more useful information can be extracted from it.

train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)
#we can also drop the Ticket feature since it's unlikely to yield any useful information

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
#now we need to fill in the missing values in the Embarked feature

print("Number of people embarking in Southampton (S):")

southampton = train[train["Embarked"] == "S"].shape[0]

print(southampton)



print("Number of people embarking in Cherbourg (C):")

cherbourg = train[train["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Number of people embarking in Queenstown (Q):")

queenstown = train[train["Embarked"] == "Q"].shape[0]

print(queenstown)
#replacing the missing values in the Embarked feature with S

train = train.fillna({"Embarked": "S"})
#create a combined group of both datasets

combine = [train, test]



#extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
#replace various titles with more common names

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train.head()
# fill missing age with mode age group for each title

mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult

miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student

mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult

master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby

royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult

rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult



age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}



#I tried to get this code to work with using .map(), but couldn't.

#I've put down a less elegant, temporary solution for now.

#train = train.fillna({"Age": train["Title"].map(age_title_mapping)})

#test = test.fillna({"Age": test["Title"].map(age_title_mapping)})



for x in range(len(train["AgeGroup"])):

    if train["AgeGroup"][x] == "Unknown":

        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

        

for x in range(len(test["AgeGroup"])):

    if test["AgeGroup"][x] == "Unknown":

        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
#map each Age value to a numerical value

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)



train.head()



#dropping the Age feature for now, might change

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)
#drop the name feature since it contains no more useful information.

train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
#map each Sex value to a numerical value

sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



train.head()
#map each Embarked value to a numerical value

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)



train.head()
#fill in missing Fare value in test set based on mean fare for that Pclass 

for x in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][x]):

        pclass = test["Pclass"][x] #Pclass = 3

        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

        

#map Fare values into groups of numerical values

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4]).values

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4]).values



#drop Fare values

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
#check train data

train.head()
#check test data

test.head()
# 시험용



import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from xgboost import plot_tree

from xgboost import plot_importance

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel

from numpy import sort
# 시험용, 테스트사이즈 0.22 -> 0.25



from sklearn.model_selection import train_test_split



predictors = train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.25, random_state = 0)

x_train.shape, x_val.shape
x_train['FareBand'].astype(int, inplace=True)
x_val['FareBand'].astype(int, inplace=True)
# xg_cl = xgb.XGBClassifier(objective='binary:logistic',learning_rate=0.01,subsample=0.55,n_estimators=200, seed=123)

clf = xgb.XGBClassifier(objective='binary:logistic',learning_rate=1)

eval_set = [(x_train.values, y_train.values), (x_val.values, y_val.values)]

# eval_set = [(x_test, y_val)]

clf.fit(x_train.values, y_train.values,early_stopping_rounds=100,eval_metric="logloss", eval_set=eval_set, verbose=True)
plot_tree(clf,num_trees=1, rankdir='LR')

plt.show()
plot_importance(clf)

plt.show()
preds = clf.predict(x_val.values)
accuracy = accuracy_score(y_val.values,preds)

accuracy
results = confusion_matrix(y_val.values, preds) 

print(results)
results = clf.evals_result()

print(results)
results = clf.evals_result()

epochs = len(results['validation_0']['logloss'])

x_axis = range(0, epochs)
tuned_params = {"objective":"binary:logistic",'colsample_bytree': 0.3, 'max_depth': 10,'subsample': 0.55, 'n_estimators': 200, 'learning_rate': 0.2}

thresholds = sort(clf.feature_importances_)

models = []

for thresh in thresholds:

    # select features using threshold

    selection = SelectFromModel(clf, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)

    # train model

    selection_model = xgb.XGBClassifier(objective='binary:logistic',learning_rate=0.8)

    selection_model.fit(select_x_train, y_train)

    # add model to models

    models.append([selection_model,selection])

    # eval model

    select_x_val = selection.transform(x_val)

    predictions = selection_model.predict(select_x_val)

    accuracy = accuracy_score(y_val, predictions)

    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_x_train.shape[1],

    accuracy*100.0))
# Finalize transformations

final_model = models[1][0]

final_selection = models[1][1]

final_x_train = final_selection.transform(x_train)



final_x_val = final_selection.transform(x_val)



final_y_pred = final_model.predict(final_x_val)

final_predictions = [round(value) for value in final_y_pred]



# Print evaluation metrics

accuracy = accuracy_score(y_val, final_predictions)

print("n=%d, Accuracy: %.2f%%" % (final_x_train.shape[1], accuracy*100.0))

confusion_matrix(y_val, final_predictions) 
kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(final_model, predictors.values, target.values, cv=kfold)

results
dmatrix = xgb.DMatrix(data=predictors.values,label=target.values)



# params={"objective":"binary:logistic","max_depth":4}

# tuned_params = {"objective":"binary:logistic",'colsample_bytree': 0.3, 'max_depth': 10,'subsample': 0.55, 'n_estimators': 200, 'learning_rate': 0.2}

tuned_params = {"objective":"binary:logistic",'learning_rate': 0.3}

tuned_params = {"objective":"binary:logistic","early_stopping_rounds":"6", "learning_rate":"0.08", "max_depth":"5", "n_estimators":"50"}



cv_results = xgb.cv(dtrain=dmatrix, params=tuned_params, nfold=5, num_boost_round=200, metrics="error",as_pandas=True, seed=123)



# Print the accuracy



print(((1-cv_results["test-error-mean"]).iloc[-1]))
# gbm_param_grid = {'learning_rate': np.arange(0.05,1.05,.05),'n_estimators': [200],'subsample': np.arange(0.05,1.05,.05)}

grid_param = {  

    'n_estimators': [12, 25, 50, 75],

    'max_depth': [3, 4, 5],

    'learning_rate': [0.01, 0.05, 0.1],

    'early_stopping_rounds': [3, 4, 5, 6]

    }

grid = RandomizedSearchCV(estimator=final_model,param_distributions=grid_param, n_iter=25,scoring='accuracy', cv=4, verbose=1)

grid.fit(predictors.values, target.values)

print("Best parameters found: ",grid.best_params_)

print("Best ROC found: ", np.sqrt(np.abs(grid.best_score_)))
x_train.head()
test.head()
selection = SelectFromModel(clf, threshold=0.012, prefit=True)

select_x_train = selection.transform(x_train)

select_x_val = selection.transform(x_val)

# another_model = xgb.XGBClassifier(early_stopping_rounds=3, learning_rate=0.5, max_depth=5, n_estimators=75)

another_model = xgb.XGBClassifier(early_stopping_rounds=5, learning_rate=0.1, max_depth=4, n_estimators=75)

another_model.fit(select_x_train, y_train)



select_y_pred = another_model.predict(select_x_val)

select_predictions = [round(value) for value in select_y_pred]



# Print evaluation metrics

accuracy = accuracy_score(y_val, select_predictions)

print("n=%d, Accuracy: %.2f%%" % (select_x_train.shape[1], accuracy*100.0))

confusion_matrix(y_val, select_predictions) 
test.isnull().all().all()

train.isnull().all().all()
train.astype(float, inplace=True)
train.info()
# train_=selection.transform(train.values)

# train_

# another_model.fit(train_,target)
x_val_=selection.transform(test.iloc[:, 1:])

prediction = another_model.predict(x_val_)
x_val_.shape
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)



# #set ids as PassengerId and predict survival 

# ids = test['PassengerId']

# predictions = gbk.predict(test.drop('PassengerId', axis=1))



# #set the output as a dataframe and convert to csv file named submission.csv

# output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

# output.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head()
# Gaussian Naive Bayes

#from sklearn.naive_bayes import GaussianNB

#from sklearn.metrics import accuracy_score



#gaussian = GaussianNB()

#gaussian.fit(x_train, y_train)

#y_pred = gaussian.predict(x_val)

#acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_gaussian)
# Logistic Regression

#from sklearn.linear_model import LogisticRegression



#logreg = LogisticRegression()

#logreg.fit(x_train, y_train)

#y_pred = logreg.predict(x_val)

#acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_logreg)
# Support Vector Machines

#from sklearn.svm import SVC



#svc = SVC()

#svc.fit(x_train, y_train)

#y_pred = svc.predict(x_val)

#acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_svc)
# Linear SVC

#from sklearn.svm import LinearSVC



#linear_svc = LinearSVC()

#linear_svc.fit(x_train, y_train)

#y_pred = linear_svc.predict(x_val)

#acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_linear_svc)
# Perceptron

#from sklearn.linear_model import Perceptron



#perceptron = Perceptron()

#perceptron.fit(x_train, y_train)

#y_pred = perceptron.predict(x_val)

#acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_perceptron)
#Decision Tree

#from sklearn.tree import DecisionTreeClassifier



#decisiontree = DecisionTreeClassifier()

#decisiontree.fit(x_train, y_train)

#y_pred = decisiontree.predict(x_val)

#acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_decisiontree)
# Random Forest

#from sklearn.ensemble import RandomForestClassifier



#randomforest = RandomForestClassifier()

#randomforest.fit(x_train, y_train)

#y_pred = randomforest.predict(x_val)

#acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_randomforest)
# KNN or k-Nearest Neighbors

#from sklearn.neighbors import KNeighborsClassifier



#knn = KNeighborsClassifier()

#knn.fit(x_train, y_train)

#y_pred = knn.predict(x_val)

#acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_knn)
# Stochastic Gradient Descent

#from sklearn.linear_model import SGDClassifier



#sgd = SGDClassifier()

#sgd.fit(x_train, y_train)

#y_pred = sgd.predict(x_val)

#acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_sgd)
# Gradient Boosting Classifier

#from sklearn.ensemble import GradientBoostingClassifier



#gbk = GradientBoostingClassifier()

#gbk.fit(x_train, y_train)

#y_pred = gbk.predict(x_val)

#acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

#print(acc_gbk)
#models = pd.DataFrame({

 #   'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

 #             'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

 #             'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

 #   'Score': [acc_svc, acc_knn, acc_logreg, 

 #             acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

 #             acc_sgd, acc_gbk]})

#models.sort_values(by='Score', ascending=False)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = gbk.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)