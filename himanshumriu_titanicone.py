import pandas as pd # data processing, CSV file I/O and Feature Engineering

import os

#ignore warnings

#import warnings

#warnings.filterwarnings('ignore')
# List all files available:

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing Training and Test data sets:

train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")



# Now Check shape of records available:

train.shape, test.shape
# There are few operation which needs to performed on both train.csv and test.csv so we pass those values by referecne to a common list variable:

data_all = [train, test]
# Function to check NULL values

def check_null(data):

    total = data.isnull().sum().sort_values(ascending=False)

    percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending=False)

    missing_data = pd.concat([total,percent], axis=1, keys=['Total', 'Percent'])

    return missing_data.head()
# Checking NULL values in Training Dataset:

check_null(train)
# Checking NULL values in test dataset:

check_null(test)
# Final Data Cleanup Operations:

col = ["Cabin", "PassengerId","Ticket"]



for dataset in data_all:

    dataset.drop(col, axis=1, inplace = True)

    dataset["Age"].fillna(dataset["Age"].median(), inplace = True)

    dataset.dropna(how='any', axis=0, inplace = True)
# Let's check again if there is any NULL records remaining in our train and test set

check_null(train)
check_null(test)
# Creating new "FamilySize" feature using already existing features:

for dataset in data_all:

    dataset["FamilySize"] = dataset["Parch"] + dataset["SibSp"] + 1
# Creating "IsAlone" feature using "FamilySize" if person is travelling alone:

for dataset in data_all:

    dataset["IsAlone"] = 1

    dataset.loc[dataset["FamilySize"] > 1,"IsAlone"] = 0



# Syntax:

# df.loc[row_index, col_index] = 0, where col is the Feature we want to update

# https://towardsdatascience.com/a-python-beginners-look-at-loc-part-2-bddef7dfa7f2
train["Sex"].value_counts()
# Feature Split

# Creating new "Title" column from "Name" column:

for dataset in data_all:

    dataset["Title"] = dataset["Name"].str.split(', ', expand = True)[1].str.split('.', expand=True)[0]
# Check Available Titles

train['Title'].value_counts()
for dataset in data_all:

    dataset.replace(to_replace=['Mlle','Ms'], value='Miss', inplace = True)

    dataset.replace(to_replace=['Mme'], value='Mrs', inplace = True)

    dataset.replace(to_replace=['Sir'], value='Mr', inplace = True)

    dataset.replace(to_replace=['Dr','Rev','Col','Major','Don','Capt','Jonkheer','the Countess','Lady','Dona'], value='Misc', inplace = True)
# Now Again Check Available Titles

train['Title'].value_counts()
# One hot Encoding of "Title", Sex" and "Embarked" feature

# This is useful because Categorical Values are challenging to understand for Algorithms.

# We will change these to Numerical values



for dataset in data_all:

    dataset['Title'] = dataset['Title'].map({'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Misc':4}).astype(int)

    dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1}).astype(int)

    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
# Now we can remove "Name" column because Title is already extracted

col = ['Name']

for dataset in data_all:

    dataset.drop(col, axis=1,inplace = True)
for dataset in data_all:

    dataset["Age"] = dataset["Age"].astype(int)

    dataset["Fare"] = dataset["Fare"].astype(int)
# Final check of our dataset

train.head()
test.head()
from matplotlib import pyplot as plt

import seaborn as sns
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score
# Feature Selection

x = train.drop('Survived', axis=1)

y = train['Survived']
# Splitting train and test dataset

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)



# Check Shape

x_train.shape,y_train.shape,x_test.shape,y_test.shape
# Creating instatnce/object of LinearRegression Class:

regressor = LogisticRegression()



# Fitting the train data sets

regressor.fit(x_train, y_train)



# Predicting values

y_predict = regressor.predict(x_test)



# Checking first 5 values

print(f'First 5 Predicted values {y_predict[:5]}.')
# Accuracy Score:

accuracy_score(y_test, y_predict)
# Importing DTree Library

from sklearn.tree import DecisionTreeClassifier



# Declaring regressor object

regressor = DecisionTreeClassifier()



# Fitting the model

regressor.fit(x_train, y_train)



# Predicting Values:

y_predict = regressor.predict(x_test)
# Checking Accuracy score

accuracy_score(y_test, y_predict)
# Importing Library

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



# Random Forest Classifier Parameters tunning 

regressor = RandomForestClassifier()

n_estim=range(100,1000,100)



## Search grid for optimal parameters

param_grid = {"n_estimators" :n_estim}





regressor_rf = GridSearchCV(regressor,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)



regressor_rf.fit(x_train,y_train)



# Best score

print(regressor_rf.best_score_)



#best estimator

regressor_rf.best_estimator_
# Now aplly the estimator which we got from above parameter tuning



regressor = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=None, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=200,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)





# Fitting the model:

regressor.fit(x_train, y_train)



# Predicting the values:

y_predict = regressor.predict(x_test)



# Checking Accuracy Score:

accuracy_score(y_test, y_predict)