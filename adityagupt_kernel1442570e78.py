import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
#read training data
titanic_train = pd.read_csv("../input/titanic/train.csv")
#read test data
titanic_test = pd.read_csv("../input/titanic/test.csv")
titanic_train.head()
titanic_train.info()
titanic_train_data = titanic_train.drop(['Cabin'],axis=1)
titanic_train_data.Age.fillna(titanic_train_data.Age.mean(), inplace=True)
titanic_train_data
titanic_train_data = titanic_train_data.dropna(axis = 0, how ='any')
titanic_train_data = titanic_train_data.drop(['Name','Ticket'],axis=1)
titanic_train_data.Sex = titanic_train_data.Sex.map({'male':0, 'female':1})
titanic_train_data.Embarked = titanic_train_data.Embarked.map({'S':0, 'C':1})
titanic_train_data = titanic_train_data.dropna(axis = 0, how ='any')
titanic_train_data['Sex'] = titanic_train_data['Sex'].astype(object).astype(int)
titanic_train_data['Embarked'] = titanic_train_data['Embarked'].astype(object).astype(int)
titanic_train_data.head()
def ageChange(a):
    if a <= 20:
        return 0
    elif a > 20 and a <= 50:
        return 1
    else:
        return 2

titanic_train_data.Age = titanic_train_data.Age.apply(ageChange)
titanic_train_data
titanic_train_data.info()
# convert target variable Survived to categorical
titanic_train_data['Survived'] = titanic_train_data['Survived'].astype('category')
# Putting feature variable to X
X = titanic_train_data.drop('Survived',axis=1)

# Putting response variable to y
y = titanic_train_data['Survived']
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state = 100)
X_train.head()
# Fitting the decision tree with default hyperparameters, apart from
# max_depth which is 5 so that we can plot and read the tree.
titanic_default = DecisionTreeClassifier(max_depth=5)
titanic_default.fit(X_train, y_train)
# Making predictions
y_pred_default = titanic_default.predict(X_test)

# Printing classification report
print(classification_report(y_test, y_pred_default))
# Printing confusion matrix and accuracy
print(confusion_matrix(y_test,y_pred_default))
print(accuracy_score(y_test,y_pred_default))
# Importing required packages for visualization
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz

# Putting features
features = list(titanic_train_data.columns[1:])
features
# plotting tree with max_depth=3
dot_data = StringIO()  
export_graphviz(titanic_default, out_file=dot_data,
                feature_names=features, filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(1, 40)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
tree.fit(X_train, y_train)
# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_leaf': range(5, 200, 20)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
tree.fit(X_train, y_train)
# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf
plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_split': range(5, 200, 20)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
tree.fit(X_train, y_train)
# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with min_samples_split
plt.figure()
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'criterion': ["entropy", "gini"]
}

n_folds = 5

# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv = n_folds, verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train,y_train)
# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results
# printing the optimal accuracy score and hyperparameters
print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)
# model with optimal hyperparameters
clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=5, 
                                  min_samples_leaf=100,
                                  min_samples_split=50)
clf_gini.fit(X_train, y_train)
# accuracy score
clf_gini.score(X_test,y_test)
# plotting the tree
dot_data = StringIO()  
export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
# classification metrics
from sklearn.metrics import classification_report,confusion_matrix
y_pred = clf_gini.predict(X_test)
print(classification_report(y_test, y_pred))
# Running the random forest with default parameters.
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
# Making predictions
predictions = rfc.predict(X_test)
# Let's check the report of our default model
print(classification_report(y_test,predictions))
# Printing confusion matrix
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))
# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(2, 20, 5)}

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
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

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
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_leaf': range(100, 400, 50)}

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
# plotting accuracies with min_samples_leaf
plt.figure()

plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [4,8,10],
    'min_samples_leaf': range(100, 400, 200),
    'min_samples_split': range(200, 500, 200),
    'n_estimators': [100,200, 300]
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
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=4,
                             min_samples_leaf=100, 
                             min_samples_split=200,
                             n_estimators=100)
# fit
rfc.fit(X_train,y_train)
# predict
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
#passenger id and final predictions
passenger_id = titanic_test['PassengerId']
test_results = pd.concat([passenger_id, pd.DataFrame(y_test)], axis=1)
test_results.columns = ['PassengerId','Survived']
test_results.dropna(axis = 0, how ='any')
