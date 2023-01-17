# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

from sklearn import svm

from sklearn import model_selection



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import train and test CSV files

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



#take a look at the training data

train_data.describe(include="all")
# Just to get an idea of the dataset

features = train_data.columns

print(features)



#printing 10 random passengers

train_data.sample(10)
# Let's check how many passengers have some NaN values



print(pd.isnull(train_data).sum())
print(train_data.shape, test_data.shape)



train_data = train_data.drop(["Cabin"], axis = 1)

test_data = test_data.drop(["Cabin"], axis = 1)



train_data = train_data.drop(["Ticket"], axis = 1)

test_data = test_data.drop(["Ticket"], axis = 1)



print(train_data.shape, test_data.shape)



features = features.drop(["Survived", "Cabin", "Ticket"])
used_features = ['Pclass', 'Sex', 'Age', "SibSp"]



final_train_data = pd.get_dummies(train_data[used_features])

labels = train_data["Survived"]



from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(final_train_data, labels, test_size = 0.20, random_state = 0)



X_test = pd.get_dummies(test_data[used_features])



print(X_train.shape, y_train.shape, X_test.shape)



X_train.sample(10)



age_mean_train = X_train.mean(axis = 0, skipna = True)["Age"] 

age_mean_val = X_val.mean(axis = 0, skipna = True)["Age"]

age_mean_test =  X_test.mean(axis = 0, skipna = True)["Age"] 



X_train["Age"].fillna(age_mean_train, inplace = True) 

X_val["Age"].fillna(age_mean_val, inplace = True) 

X_test["Age"].fillna(age_mean_test, inplace = True) 



# Checking if NaN count of age feature is 0

print(pd.isnull(X_train).sum())

print(pd.isnull(X_val).sum())



training_scores = []

validation_scores = []
svc = svm.SVC()

parameters = {'kernel':['rbf'], 'C': [0.1, 0.01, 1, 10, 100], 'gamma':[0.01, 0.1,1.,10, 100]}

svm_clf = model_selection.GridSearchCV(svc, parameters, cv = 5, n_jobs = -1, verbose = True)

svm_clf.fit(X_train, y_train)

final_svm = svm_clf.best_estimator_



from sklearn.metrics import classification_report

print(classification_report(y_train, final_svm.predict(X_train)))

svm_training_score = final_svm.score(X_train, y_train)

training_scores.append(svm_training_score)

print("The score is %f !" %(svm_training_score))
from sklearn.linear_model import LogisticRegressionCV



log_reg = LogisticRegressionCV([0.1, 0.01, 1, 10, 100], max_iter = 10000, n_jobs = -1, cv=5, verbose = True).fit(X_train, y_train)

print(classification_report(y_train, log_reg.predict(X_train)))



log_reg_training_score = log_reg.score(X_train, y_train)

training_scores.append(log_reg_training_score)

print("The score is %f !" %(log_reg_training_score))
from sklearn.neural_network import MLPClassifier



parameters = {'hidden_layer_sizes': [(4,), (10,), (50,), (10,10,), (50,50,), (4,4), (10,10,10,10)]} # "activation" : ['identity', 'logistic', 'tanh', 'relu']}

nn_cv = model_selection.GridSearchCV(MLPClassifier(max_iter=1000, random_state = 0), parameters, n_jobs = -1, cv = 5).fit(X_train, y_train)



print(nn_cv.cv_results_)



print(classification_report(y_train, nn_cv.predict(X_train)))



nn_cv_training_score = nn_cv.score(X_train, y_train)

training_scores.append(nn_cv_training_score)

print("The score is %f !" %(nn_cv_training_score))
from sklearn.tree import DecisionTreeClassifier

parameters = {"criterion": ["gini", "entropy"], 'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}

tree_cv = model_selection.GridSearchCV(DecisionTreeClassifier(), parameters,  cv = 5, n_jobs = -1, verbose = True).fit(X_train, y_train)

print(classification_report(y_train, tree_cv.predict(X_train)))



tree_cv_training_score = tree_cv.score(X_train, y_train)

training_scores.append(tree_cv_training_score)

print("The score is %f !" %(tree_cv_training_score))
from sklearn.ensemble import RandomForestClassifier

parameters = {"criterion": ["gini", "entropy"], 'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}

rand_forest_cv = model_selection.GridSearchCV(RandomForestClassifier(), parameters, n_jobs = -1, cv = 5, verbose = True).fit(X_train, y_train)



print(classification_report(y_train, rand_forest_cv.predict(X_train)))



rand_forest_cv_training_score = rand_forest_cv.score(X_train, y_train)

training_scores.append(rand_forest_cv_training_score)

print("The score is %f !" %(rand_forest_cv_training_score))
validation_scores = [final_svm.score(X_val, y_val), log_reg.score(X_val, y_val), nn_cv.score(X_val, y_val), tree_cv.score(X_val, y_val), rand_forest_cv.score(X_val, y_val)]



print("Validation score of SVM Classifier: %f" %(validation_scores[0]))

print("Validation score of Logistic Regression Classifier: %f" %(validation_scores[1]))

print("Validation score of Neural Network Classifier: %f" %(validation_scores[2]))

print("Validation score of Tree Classifier: %f" %(validation_scores[3]))

print("Validation score of Random Forest Classifier: %f" %(validation_scores[4]))



data = []

for element in list(zip(training_scores, validation_scores)):

      data.append({"Training score": element[0], "Validation score": element[1]})

      

results = pd.DataFrame(data, index = ["SVM RBF", "Logistic Regression", "Neural Network", "Decision Tree", "Random Forest"])

print(results)
predictions = rand_forest_cv.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")