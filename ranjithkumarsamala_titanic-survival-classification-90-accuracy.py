# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import scipy as sp

import sklearn 



# misc libraries

import random

import time



#ignore warnings 

import warnings

warnings.filterwarnings("ignore")



##Data modelling libraries## 

#Common model algorithms 

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis,gaussian_process

from xgboost import XGBClassifier 



#Common models

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection 

from sklearn import metrics



#visualization libraries

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.plotting import scatter_matrix

import missingno



#some more visulization configuarations

%matplotlib inline

mpl.style.use("ggplot")

sns.set_style("white")

pylab.rcParams["figure.figsize"]=12,8



# Start Python Imports

import math, time, random, datetime



# Data Manipulation

import numpy as np

import pandas as pd



# Visualization 

import matplotlib.pyplot as plt

import missingno

import seaborn as sns



# Preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

from sklearn.preprocessing import StandardScaler, RobustScaler



# Machine learning

import catboost

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import SVC

from sklearn.linear_model import Perceptron

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier, Pool, cv

# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,roc_curve, auc



#model to pickel format 

import pickle

# Let's be rebels and ignore warnings for now

import warnings

warnings.filterwarnings('ignore')

train_raw = pd.read_csv("../input/titanic/train.csv")
test_raw = pd.read_csv("../input/titanic/test.csv")
# copying train data for preparing 

data = pd.DataFrame.copy(train_raw)
#making two datasets together to clear data

data_cleaner = [data, test_raw]
data_cleaner
print(train_raw.info(),train_raw.columns)
data.describe(include="all")
#custom function checking for missing values

def missingdata(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    missing_data=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    missing_data= missing_data[missing_data["Percent"] > 0]

    f,ax =plt.subplots(figsize=(8,6))

    plt.xticks(rotation='90')

    fig=sns.barplot(missing_data.index, missing_data["Percent"],color="green",alpha=0.8)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)

    return missing_data 
#checking missing values in train data

missingdata(data)
#checking missing values in test data

missingdata(test_raw)
# replacing missing values with mean, median & mode in respective variables

# for loop because list is not callable 



for dataset in data_cleaner:

    #replacing with median

    dataset["Age"].fillna(dataset["Age"].median(), inplace=True)

    #replacing with median

    dataset["Fare"].fillna(dataset["Fare"].median(), inplace=True)

    #replacing with mode

    dataset["Embarked"].fillna(dataset["Embarked"].mode()[0], inplace=True)
#checking missing values after imputation

missingno.matrix(data, figsize = (30,10))

# drop paasnger ID, cabin, ticket from train1

drop_column = ["PassengerId","Cabin", "Ticket"]

data.drop(drop_column, axis=1, inplace=True)
# making new variables to define family members from SibSp & Parch

# for continous variables (age & fare), making new discreted variables

for dataset in data_cleaner:    

    # creating "FamilySize" variable based on "SibSp & Parch" data 

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    # creating "IsAlone" variable with 0s(not alone) and 1s(alone)  

    dataset['IsAlone'] = 1 

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 

    # seperating split title and name from name

    dataset['Title'] = dataset['Name'].str.split(",",expand=True)[1].str.split(".",expand=True)[0]

    #making continous variables into discete variables 

    dataset['FareBin'] = pd.cut(dataset['Fare'],4)

    dataset['AgeBin'] = pd.cut(dataset['Age'],5)
# discreting title variable by count (min count is 10)

# saving below 10 unique name tiltes under "misc" title

misc_min_range = 10 

title_names = (data['Title'].value_counts()<misc_min_range)

# this will create a true or false series with title name as index



data['Title'] = data['Title'].apply(lambda x:'Misc' if title_names.loc[x]==True else x)

print(data['Title'].value_counts())

print("-"*25)
data.head(10)
# segregation of titles

title_names = (test_raw['Title'].value_counts()<misc_min_range)

# this will create a true or false series with title name as index



test_raw['Title'] = test_raw['Title'].apply(lambda x:'Misc' if title_names.loc[x]==True else x)

print(test_raw['Title'].value_counts())

print("-"*25)
# converting objects to categorical data using labelencoder

label = LabelEncoder()

for dataset in data_cleaner:

    dataset["Sex_Code"] = label.fit_transform(dataset["Sex"])

    dataset["Embarked_Code"] = label.fit_transform(dataset["Embarked"])

    dataset["Title_Code"] = label.fit_transform(dataset["Title"])

    dataset["FareBin_Code"] = label.fit_transform(dataset["FareBin"])

    dataset["AgeBin_Code"] = label.fit_transform(dataset["AgeBin"])
data.head()
test_raw.head()
# segregating data for irrespective use



# for model building

col_model = ["Name","Age","Fare","FareBin","AgeBin","Sex_Code","Embarked_Code","Title_Code"]

titan_model = pd.DataFrame.copy(data)

titan_model.drop(col_model, axis=1, inplace=True)

titan_model.head()
# for EDA charts

col_EDA = ["Name","Title_Code","FareBin","AgeBin","Sex_Code","Embarked_Code"]

titan_EDA = pd.DataFrame.copy(data)

titan_EDA.drop(col_EDA, axis=1, inplace=True)

titan_EDA.head()
# creating dummy variables for train data (titan_model)

titan_model = pd.get_dummies(titan_model, columns = ["Sex","Embarked","Title","FareBin_Code","AgeBin_Code"],

                             prefix=["Sex","Embarked_type","Title","Fare_type","Age_type"])
titan_model.head()
x = titan_model.drop('Survived', axis=1)

y = titan_model['Survived']
# test data for prediction

col_delete = ["PassengerId","Name","Age","Ticket","Fare","Cabin","FareBin","AgeBin","Sex_Code","Embarked_Code","Title_Code" ]

test_model = pd.DataFrame.copy(test_raw)

test_model.drop(col_delete, axis=1, inplace=True)

test_model.head()
# creating dummy variables for test data (test_model)

test_model = pd.get_dummies(test_model, columns = ["Sex","Embarked","Title","FareBin_Code","AgeBin_Code"],

                             prefix=["Sex","Embarked_type","Title","Fare_type","Age_type"])
test_model.head()
print((titan_model.columns,test_model.columns))
# percentage ratio of Survived and Not Survived

colors = ["#61d4b3", "#ff2e63"]

sns.countplot('Survived', data=titan_EDA, palette=colors)

plt.title('Survival Distribution', fontsize=20)

plt.xlabel('1: Survived, 0: Not Survived')

plt.show()

# AGE vs SURVIVAL

g = sns.kdeplot(titan_EDA["Age"][(titan_EDA["Survived"] == 0) & (titan_EDA["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(titan_EDA["Age"][(titan_EDA["Survived"] == 1) & (titan_EDA["Age"].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
# Boxplots AGE vs SEX & Family

sns.factorplot(y="Age",x="Sex",data=titan_EDA,kind="box")

sns.factorplot(y="Age",x="Sex",hue="Pclass", data=titan_EDA,kind="box")

sns.factorplot(y="Age",x="Parch", data=titan_EDA,kind="box")

sns.factorplot(y="Age",x="SibSp", data=titan_EDA,kind="box")
# Display all possible features with Survival

colors1 = ["#00a8cc", "#54123b"]

possible_features = ['Pclass', 'Sex', 'SibSp', 'Parch','Embarked', 'FamilySize', 'IsAlone', 'Title', 'FareBin_Code','AgeBin_Code']

fig, axs = plt.subplots(5, 2, figsize=(20, 30))

for i in range(0, 10):

    sns.countplot(possible_features[i], data=titan_EDA, palette=colors1, hue="Survived", ax=axs[i%5, i//5])

titan_EDA.columns
# Heat map corelation

sns.heatmap(titan_EDA[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")

plt.subplot(231)

plt.hist(x = [titan_EDA[titan_EDA['Survived']==1]['Age'], titan_EDA[titan_EDA['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Not survived'])

plt.title('Age Histogram with Survival')

plt.xlabel('Age')

plt.ylabel('# No of passengers')

plt.legend()



plt.subplot(232)

plt.hist(x = [titan_EDA[titan_EDA['Survived']==1]['Fare'], titan_EDA[titan_EDA['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Not survived'])

plt.title('Fare Histogram with Survival')

plt.xlabel('Fare')

plt.ylabel('# No of passengers')

plt.legend()
# Our data is already scaled we should split our training and test sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33,stratify=y,random_state =42)



# Turn the values into an array for feeding the classification algorithms.

x_train = x_train.values

x_test = x_test.values

y_train = y_train.values

y_test = y_test.values
classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "Naive Bayes Classifier": GaussianNB(),

    "DecisionTreeClassifier": DecisionTreeClassifier(),

    "RandomForestClassifier": RandomForestClassifier(),

    "Stochastic Gradient Descent": SGDClassifier(),

}



for key, classifier in classifiers.items():

    classifier.fit(x_train, y_train)

    training_score = cross_val_score(classifier, x_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")



# Use GridSearchCV to find the best parameters.

# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(x_train, y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_
# KNN

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(x_train, y_train)

# KNears best estimator

knears_neighbors = grid_knears.best_estimator_
# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(x_train, y_train)

# SVC best estimator

svc = grid_svc.best_estimator_
# Naive and Bayes classifier

gnb_params = {}

grid_gnb = GridSearchCV(GaussianNB(), gnb_params)

grid_gnb.fit(x_train, y_train)

# nb best estimator

gnb = grid_gnb.best_estimator_

# DecisionTree Classifier

tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 

              "min_samples_leaf": list(range(5,35,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(x_train, y_train)

# tree best estimator

tree_clf = grid_tree.best_estimator_
# RandomForest Classifier

rfc_params = {"n_estimators":list(range(1,100,1)), "criterion": ["gini", "entropy"]}

grid_rfc = GridSearchCV(RandomForestClassifier(), rfc_params)

grid_rfc.fit(x_train, y_train)

# rf best estimator

rfc = grid_rfc.best_estimator_
#  SGDClassifier

sgd_params = {"max_iter":list(range(1,5,1))}

grid_sgd = GridSearchCV(SGDClassifier(), sgd_params)

grid_sgd.fit(x_train, y_train)

# sgd best estimator

sgd = grid_sgd.best_estimator_
# Overfitting Case

log_reg_score = cross_val_score(log_reg, x_train, y_train, cv=5)

print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')



knears_score = cross_val_score(knears_neighbors, x_train, y_train, cv=5)

print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')



svc_score = cross_val_score(svc, x_train, y_train, cv=5)

print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')



gnb_score = cross_val_score(gnb, x_train, y_train, cv=5)

print('Naive Bayes Cross Validation Score', round(gnb_score.mean() * 100, 2).astype(str) + '%')



tree_score = cross_val_score(tree_clf, x_train, y_train, cv=5)

print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')



rfc_score = cross_val_score(rfc, x_train, y_train, cv=5)

print('Random Forest Classifier Cross Validation Score', round(rfc_score.mean() * 100, 2).astype(str) + '%')



sgd_score = cross_val_score(sgd, x_train, y_train, cv=5)

print('SGD Classifier Cross Validation Score', round(sgd_score.mean() * 100, 2).astype(str) + '%')
# Building Neural Network model also

features = x.values

target = y.values
# libraries for NN

import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import cross_val_score

from keras.wrappers.scikit_learn import KerasClassifier
classifier = Sequential()

classifier.add(Dense(activation="relu", input_dim=24, units=12, kernel_initializer="uniform"))

classifier.add(Dense(activation="relu", units=12, kernel_initializer="uniform"))

classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(features, target, batch_size = 10, nb_epoch = 100)
def build_classifier(optimizer):

    classifier = Sequential()

    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 24))

    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],

              'epochs': [100,500],

              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10)

grid_search = grid_search.fit(features, target)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_
[best_parameters, best_accuracy]
# building final model based on above parameters

classifier = Sequential()

classifier.add(Dense(activation="relu", input_dim=24, units=12, kernel_initializer="uniform"))

classifier.add(Dense(activation="relu", units=12, kernel_initializer="uniform"))

classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model_history=classifier.fit(features, target, validation_split=0.33, batch_size = 25, nb_epoch = 100)

print(model_history.history.keys())
# summarize history for accuracy

plt.plot(model_history.history['accuracy'])

plt.plot(model_history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred = classifier.predict(test_model)
y_pred.dtype
#Round off the result for submission

y_pred=y_pred.round()
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
gender_submission['Survived'] = y_pred
gender_submission.to_csv("../working/submit.csv", index=False)