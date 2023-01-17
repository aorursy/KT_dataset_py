# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# First Step: Get the Data

AUTOPATH = "../input"



def load_titanic_training_data(data_path=AUTOPATH):

    csv_path = os.path.join(AUTOPATH,"train.csv")

    return pd.read_csv(csv_path)



training_data = load_titanic_training_data()

training_data.head(3)
training_data.info()
training_data["Sex"].value_counts()
training_data["Cabin"].value_counts()
training_data["Ticket"].value_counts()
training_data["Embarked"].value_counts()
training_data.describe()
# Setting up graphs to visually display the data for interpretation and modification

%matplotlib inline

import matplotlib.pyplot as plt

training_data.hist(bins=50, figsize=(30,20))
# Checking for correlation between variables

corr_matrix = training_data.corr()

corr_matrix["Survived"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix



attributes = ["Fare", "Pclass", "Age", "Parch", "SibSp"]

scatter_matrix(training_data[attributes], figsize=(30,20))
# Separating target label from training data

training_label = training_data["Survived"].copy()

training_data = training_data.drop("Survived", axis=1)

training_data.head(5)
# Removing columns not in attributes

non_attributes = ["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]

training_data.drop(non_attributes, axis=1, inplace=True)
# Addressing the missing values in training data

training_data.isnull().sum()
# Filling in missing data with SimpleImputer

from sklearn.impute import SimpleImputer

imputer_med = SimpleImputer(strategy="median")

imputer_mea = SimpleImputer(strategy="mean")

imputer_mod = SimpleImputer(strategy="most_frequent")



# Filled in data w/ median

imputer_med.fit(training_data.drop("Sex", axis=1))

print(imputer_med.statistics_)

# Filled in data w/ mean

imputer_mea.fit(training_data.drop("Sex", axis=1))

print(imputer_mea.statistics_)

# Filled in data w/ mode

imputer_mod.fit(training_data.drop("Sex", axis=1))

print(imputer_mod.statistics_)
training_data.info()
# Filling in missing values of age with the mean age

mean_age = imputer_mea.statistics_[1]

training_data["Age"].fillna(mean_age, inplace=True)
# Encoding Sex attribute into 0s and 1s by using dict replace

training_data_sex = training_data[["Sex"]]



sex_dict = {"male": 0, "female": 1}

training_data.replace(sex_dict, inplace=True)



training_data.head(5)
# Setting up confusion matrix, precision, recall, f1score, roc auc score

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score



def print_metrics(labels, predictions):

    print("Confusion Matrix:\n", confusion_matrix(labels, predictions))

    print("Precision:", precision_score(labels, predictions))

    print("Recall:", recall_score(labels, predictions))

    print("F1 Score:", f1_score(labels, predictions))

    print("ROC AUC Score:", roc_auc_score(labels, predictions))

    print("Accuracy:", accuracy_score(labels, predictions))
# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier



dtclf = DecisionTreeClassifier(random_state=42)

dtclf.fit(training_data, training_label)



# Doing cross validation predictions for Decision Tree Classifier

from sklearn.model_selection import cross_val_predict

dtclf_preds = cross_val_predict(dtclf, training_data, training_label, cv=3)



# dtclf metrics

print_metrics(training_label, dtclf_preds)
# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier



rfclf = RandomForestClassifier(random_state=42)

rfclf.fit(training_data, training_label)



# Cross validation predictions

rfclf_preds = cross_val_predict(rfclf, training_data, training_label, cv=3)



# rfclf metrics

print_metrics(training_label, rfclf_preds)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB



gnbclf = GaussianNB()

gnbclf.fit(training_data, training_label)



# Cross validation predictions

gnbclf_preds = cross_val_predict(gnbclf, training_data, training_label, cv=3)



# gnbclf metrics

print_metrics(training_label, gnbclf_preds)
# Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB



mnbclf = MultinomialNB()

mnbclf.fit(training_data, training_label)



# Cross validation predictions

mnbclf_preds = cross_val_predict(mnbclf, training_data, training_label, cv=3)



# mnbclf metrics

print_metrics(training_label, mnbclf_preds)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbcclf = GradientBoostingClassifier(n_estimators=200, random_state=42)

gbcclf.fit(training_data, training_label)



# Cross validation predictions

gbcclf_preds = cross_val_predict(gbcclf, training_data, training_label, cv=3)



# gbcclf metrics

print_metrics(training_label, gbcclf_preds)
# Ada Boost Classifier

from sklearn.ensemble import AdaBoostClassifier



abcclf = AdaBoostClassifier(n_estimators=200, random_state=42)

abcclf.fit(training_data, training_label)



# Cross validation predictions

abcclf_preds = cross_val_predict(abcclf, training_data, training_label, cv=3)



# abcclf metrics

print_metrics(training_label, abcclf_preds)
# KNeighbors Classifier

from sklearn.neighbors import KNeighborsClassifier



kncclf = KNeighborsClassifier(n_neighbors=5)

kncclf.fit(training_data, training_label)



# Cross validation predictions

kncclf_preds = cross_val_predict(kncclf, training_data, training_label, cv=3)



# kncclf metrics

print_metrics(training_label, kncclf_preds)
# SVM Classifier

from sklearn.svm import SVC



svcclf = SVC(gamma='scale', random_state=42)

svcclf.fit(training_data, training_label)



# Cross validation predictions

svcclf_preds = cross_val_predict(svcclf, training_data, training_label, cv=3)



# svcclf metrics

print_metrics(training_label, svcclf_preds)
# Neural Network: Multilayer Perceptron Classifier

from sklearn.neural_network import MLPClassifier



mlpcclf = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)

mlpcclf.fit(training_data, training_label)



# Cross validation predictions

mlpcclf_preds = cross_val_predict(mlpcclf, training_data, training_label, cv=3)



# mlpcclf metrics

print_metrics(training_label, mlpcclf_preds)
# GridSearchCV for best features

from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}

]



# May need to do separate GridSearchCV and RandomizedSearchCV for each individual model

# Should try to automate this process
# Submission to competition

def load_titanic_testing_data(data_path=AUTOPATH):

    csv_path = os.path.join(AUTOPATH,"test.csv")

    return pd.read_csv(csv_path)



# Loading test data

testing_data = load_titanic_testing_data()



# Saving ids of passengers in testing data (need it later for submission)

testing_ids = testing_data["PassengerId"].copy()



# Removing features not used in training data

testing_data.drop(non_attributes, axis=1, inplace=True)



# Filling in missing values in testing data Age column

testing_data_imputer = SimpleImputer(strategy="mean")

testing_data_imputer.fit(testing_data.drop("Sex", axis=1))



testing_mean_age = testing_data_imputer.statistics_[1]

testing_data["Age"].fillna(testing_mean_age, inplace=True)



# Filling in missing value in Fare column

testing_mean_fare = round(testing_data["Fare"].mean(), 4)

testing_data["Fare"].fillna(testing_mean_fare, inplace=True)



# Encoding Sex column w/ binary

testing_data.replace(sex_dict, inplace=True)



# Gradient Boosting Classifier has highest score, so we will use that model

submission_preds = gbcclf.predict(testing_data)



output = pd.DataFrame({'PassengerId': testing_ids,

                       'Survived': submission_preds})

output.to_csv('gender_submission.csv', index=False)