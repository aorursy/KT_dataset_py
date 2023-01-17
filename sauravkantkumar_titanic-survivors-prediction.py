# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting

import seaborn as sns

from sklearn.impute import SimpleImputer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# Loading the Train and Test into Pandas DataFrames. After pre-processing it can be converted to H2OFrame.

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Look at the initial rows of train data

train.head(5)
#Look at train data statistics

train.describe()
# Check the number of null values in each column of train DataFrame

train.isnull().sum()
#Plot the number of null values in every column.

plt.figure(figsize=[10, 5])

sns.heatmap(train.isnull())
# Look at the initial rows of test data

test.head(5)
#Look at test data statistics

test.describe()
# Check the number of null values in each column of test DataFrame

test.isnull().sum()
#Plot the number of null values in every column.

plt.figure(figsize=[10, 5])

sns.heatmap(test.isnull())
# Lets see how much percentage of people survived

print(train.Survived.value_counts()*100/train.shape[0])

# Here, we can see that only 38.38 percent people survived
# count plot for Survived column

plt.figure(figsize=[10, 5])

sns.countplot(x='Survived',data=train)

# Below graph shows that number of survivers are less compared to non-survivers.
# Lets see how many male and female Survived  

plt.figure(figsize=[10, 5])

sns.countplot(x='Survived',data=train, hue='Sex')

# Below graph shows that more number of females survived than males
# Lets see how many survivers are there from each class 

plt.figure(figsize=[10, 5])

sns.countplot(x='Survived',data=train, hue='Pclass')

# Below graph shows that maximum number of survivers were from class 1 and  

# the maximum casualties were from class 3
plt.figure(figsize=[10, 6])

sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train)

plt.show()

# Below figure describes that among those who survived, number of female surviors are more in

# each of the Pclass. 
from sklearn.impute import SimpleImputer

imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
train['Age'] = imp_median.fit_transform(train['Age'].values.reshape(-1,1))
train.isnull().sum()
# Dropping Cabin column as there are lot of missing values

train.drop('Cabin',axis=1,inplace=True)
train.isnull().sum()
train = train.dropna()
train.isnull().sum()
plt.figure(figsize=[10, 5])

sns.heatmap(train.isnull())

# There are no null values left in the train DataFrame.
# Dropping the Columns PassengerId,Name,Ticket from the train DataFrame as they are specific to each passenger

# and do not add any value to our analysis.

train = train.drop(['PassengerId','Name','Ticket'], axis=1)
train.columns
# Putting feature variable to X

X_train = train.drop('Survived',axis=1)

# Putting response variable to y

y_train = train['Survived']
test.isnull().sum()
test.head()
# Imputing the missing age value with the median value

test['Age'] = imp_median.fit_transform(test['Age'].values.reshape(-1,1))
test.shape
test.isnull().sum()
# Dropping the Cabin column as it contain large number of missing values.

test.drop('Cabin',axis=1,inplace=True)
test.isnull().sum()
# Check if there is any null value left in any column of test DataFrame

plt.figure(figsize=[10, 5])

sns.heatmap(test.isnull())
test['Fare'].median()
test['Fare'].fillna(value=15,inplace=True)
# Check if there is any null value left in any column of test DataFrame

plt.figure(figsize=[10, 5])

sns.heatmap(test.isnull())
# Putting feature variable to X

X_test = test.drop(['PassengerId', 'Name','Ticket'], axis=1)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
train['Sex'].dtype
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X_train["Sex"] = le.fit_transform(X_train["Sex"])

X_train["Embarked"] = le.fit_transform(X_train["Embarked"])

X_test["Sex"] = le.fit_transform(X_test["Sex"])

X_test["Embarked"] = le.fit_transform(X_test["Embarked"])
X_train.columns
X_train['Embarked'].dtype
X_train.head(5)
# Importing random forest classifier from sklearn library

from sklearn.ensemble import RandomForestClassifier



# Running the random forest with default parameters.

rfc = RandomForestClassifier()
# fit

rfc.fit(X_train,y_train)
y_predict_default = rfc.predict(X_test)
# GridSearchCV to find optimal n_estimators

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(2, 20, 2)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_depth

plt.figure()

plt.plot(scores["param_max_depth"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_depth"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_depth")

plt.ylabel("accuracy")

plt.legend()

plt.show()

# GridSearchCV to find optimal n_estimators

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'n_estimators': range(100, 1500, 400)}



# instantiate the model (note we are specifying a max_depth)

rf = RandomForestClassifier(max_depth=4)





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with n_estimators

plt.figure()

plt.plot(scores["param_n_estimators"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_n_estimators"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("n_estimators")

plt.ylabel("Accuracy")

plt.legend()

plt.show()

# GridSearchCV to find optimal max_features

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_features': range(2, 8, 1)}



# instantiate the model

rf = RandomForestClassifier(max_depth=4)





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with max_features

plt.figure()

plt.plot(scores["param_max_features"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_max_features"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("max_features")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# GridSearchCV to find optimal min_samples_leaf

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(2, 20, 1)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf

plt.figure()

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_min_samples_leaf"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_leaf")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# GridSearchCV to find optimal min_samples_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_split': range(2, 100, 2)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

rf.fit(X_train, y_train)
# scores of GridSearch CV

scores = rf.cv_results_

pd.DataFrame(scores).head()
# plotting accuracies with min_samples_split

plt.figure()

plt.plot(scores["param_min_samples_split"], 

         scores["mean_train_score"], 

         label="training accuracy")

plt.plot(scores["param_min_samples_split"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("min_samples_split")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [4,6,8,10],

    'min_samples_leaf': [13, 14, 15],

    'min_samples_split': [38,39,40,41,42,43],

    'n_estimators': [400,500, 600], 

    'max_features': [4, 5, 6]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1,verbose = 1)
# Fit the grid search to the data

grid_search.fit(X_train, y_train)
# printing the optimal accuracy score and hyperparameters

print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)
# model with the best hyperparameters

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(bootstrap=True,

                             max_depth=4,

                             min_samples_leaf=13, 

                             min_samples_split=43,

                             max_features=6,

                             n_estimators=400)
# fit

rfc.fit(X_train,y_train)
Y_predict = rfc.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_predict

    })

submission.to_csv('titanic.csv', index=False)