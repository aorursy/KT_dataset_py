#import the pandas library.

#we need to handle with arrays. So, we import numpy library as well.

import pandas as pd

import numpy as np



train = pd.read_csv('https://raw.githubusercontent.com/gurusabarish/Titanic-prediction/master/train.csv')

test = pd.read_csv('https://raw.githubusercontent.com/gurusabarish/Titanic-prediction/master/test.csv')
#Print the first few rows of train dataset.

train.head()
# we dont use name and Ticket columns. So, we can drop those columns.

train.drop(['Name', 'Ticket'],axis=1, inplace=True)
#find the number of rows and columns

train.shape
#print the columns with his datatype and non-null(filled) counts.

train.info()
# We should few columns in test data as well as train dta

test.drop(['Name', 'Ticket'],axis=1, inplace=True)
#import libraries

# we will find "how many peoples are have the same value in respective column?"



import matplotlib.pyplot as plt

train.hist(figsize=(20,10), color='maroon', bins=20)

plt.show()
# Count of missing data in each columns

train.isnull().sum()
# Find the percentage of missing data with respective columns.

columns = train.columns

for i in columns:

    percentage = (train[i].isnull().sum()/891)*100

    print(i,"\t %.2f" %percentage)
#Cabin has almost 80% missing data.So, we are drop the column.

train.drop(['Cabin'], axis=1, inplace=True)
#Embrked column has only few NAN value.So, drop the row which is contain the missing data.

train.dropna(subset=['Embarked'], inplace=True)
# Fill the NAN values with mean in Age column

train['Age'].fillna(train['Age'].mean(), inplace=True)
# Check the columns are don't contain the NAN values.

train.isnull().any()
test.drop(['Cabin'], axis=1, inplace=True)
test.isnull().sum()
test['Age'].fillna(train['Age'].mean(), inplace=True)

test['Fare'].fillna(train['Fare'].mean(), inplace=True)
test.isnull().any()
# Import the function from sklearn library

from sklearn import preprocessing

label = preprocessing.LabelEncoder()
# Find out the columns which is have object data type and store the column names.

object_columns = []

for i in train.columns:

    if train[i].dtype==object:

        object_columns.append(i)

print(object_columns)
# Convert the columns's object data to numerical data.

for i in object_columns:

    train[i] = label.fit_transform(train[i])
# Verify the dataset is contain only numerical values.

train.info()
test_object_columns = []

for i in test.columns:

    if test[i].dtype==object:

        test_object_columns.append(i)

print(test_object_columns)
for i in test_object_columns:

    test[i] = label.fit_transform(test[i])
test.info()
x = train.drop('Survived', axis=1)

y = train['Survived']
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB 

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
cross_val_score(LogisticRegression(), x, y).mean()
cross_val_score(SVC(), x, y).mean()
cross_val_score(RandomForestClassifier(), x, y).mean()
cross_val_score(GaussianNB(), x, y).mean()
cross_val_score(DecisionTreeClassifier(), x, y).mean()
# Split the train dataset to build model.

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.30, random_state=5)
from sklearn.metrics import confusion_matrix,accuracy_score

def evaluate(model, x_test, y_test):

  prediction = model.predict(x_test)

  print(accuracy_score(y_test, prediction))
from sklearn.model_selection import GridSearchCV# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [4,6,8,10,12],

    'max_features': ['auto', 'sqrt', 'log2'],

    'min_samples_leaf': [1, 3, 4, 5],

    'min_samples_split': [1, 2, 4, 5],

    'n_estimators': [100, 200, 300, 1000]

}# Create a based model

rf = RandomForestClassifier()# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data

grid_search.fit(x_train, y_train)

best = grid_search.best_params_
grid_search.best_estimator_
model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=12, max_features='auto',

                       max_leaf_nodes=None, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=5,

                       min_weight_fraction_leaf=0.0, n_estimators=200,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)

model.fit(x_train,y_train)
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix,accuracy_score



predictions = model.predict(x_test)

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))

print(accuracy_score(y_test, predictions))
test.head()
survived=model.predict(test)

survived
ids = test['PassengerId']

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': survived })

output.head()
output.to_csv('output.csv', index=False)