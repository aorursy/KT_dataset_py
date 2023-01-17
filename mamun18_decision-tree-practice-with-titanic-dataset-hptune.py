# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings('ignore')
train_path = '/kaggle/input/titanic/train.csv'
test_path = '/kaggle/input/titanic/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
train_df.head()
test_df.head()
train_df.info()
train_df.describe()
train_df.shape
train_df.isnull().sum()
sns.set()
sns.countplot(train_df['Embarked']);
# filling the missing value of Embarked with S because S is frequently used
train_df = train_df.fillna({"Embarked": "S"})
train_df['Embarked'].isnull().sum()
# One hot encoding applied on Embarked column(no ordering)
train_df = pd.get_dummies(train_df, columns = ['Embarked'])
train_df.head()

sns.set()
sns.countplot(train_df['Sex']);
# One hot encoding applied on sex column
train_df = pd.get_dummies(train_df, columns = ['Sex'])
train_df.head()
features = ['Pclass', 'Sex_male', 'Sex_female', 'Embarked_C', 'Embarked_Q',
           'Embarked_S', 'Parch', 'SibSp', 'Fare']

train_label = train_df[['Survived']]
train_features = train_df[features]
train_features.head()
train_features.isnull().sum()
# Importing decision tree classifier from sklearn library
from sklearn.tree import DecisionTreeClassifier

# Fitting the decision tree with default hyperparameters, apart from
# max_depth which is 5 so that we can plot and read the tree.
dt_default = DecisionTreeClassifier(max_depth=5, random_state=0)
dt_default.fit(train_features, train_label)
# Let's check the evaluation metrics of our default model

# Importing classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Making predictions
y_pred_default = dt_default.predict(train_features)

# Printing classification report
print(classification_report(train_label, y_pred_default))
# Printing confusion Matrix and accuracy
print(confusion_matrix(train_label, y_pred_default))
print(accuracy_score(train_label, y_pred_default))
from sklearn import tree
plt.figure(figsize=(10, 8))
tree.plot_tree(dt_default)
# Importing required packages for visualization
from sklearn.tree import export_graphviz
import graphviz

features
data = export_graphviz(dt_default, out_file=None, feature_names = features,
                       class_names = 'Survived', filled = True,
                      rounded = True, special_characters = True)
graph = graphviz.Source(data)
graph
# GridSearchCV to find optimal max_depth
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Specify number of folds for KFolds CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(1, 40)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion='gini',
                              random_state = 100)
# fit tree on training data
tree = GridSearchCV(dtree, parameters,
                   cv = n_folds,
                   scoring = 'accuracy')
tree.fit(train_features, train_label)
# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()
# plotting accuracy with max_depth
plt.figure()
plt.plot(scores['param_max_depth'],
        scores['mean_test_score'],
        label=('Training accuracy'))

plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# GridSearchCV to find optimal min_samples_leaf
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Specify number of folds for KFolds CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_leaf': range(5, 200, 20)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion='gini',
                              random_state = 100)
# fit tree on training data
tree = GridSearchCV(dtree, parameters,
                   cv = n_folds,
                   scoring = 'accuracy')
tree.fit(train_features, train_label)
# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()
# plotting accuracy with min_samples_leaf
plt.figure()
plt.plot(scores['param_min_samples_leaf'],
        scores['mean_test_score'],
        label=('Training accuracy'))

plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# GridSearchCV to find optimal min_samples_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Specify number of folds for KFolds CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_split': range(5, 200, 20)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion='gini',
                              random_state = 100)
# fit tree on training data
tree = GridSearchCV(dtree, parameters,
                   cv = n_folds,
                   scoring = 'accuracy')

tree.fit(train_features, train_label)
# scores of GridSearch CV
scores = tree.cv_results_
pd.DataFrame(scores).head()
# plotting accuracy with min_samples_split
plt.figure()
plt.plot(scores['param_min_samples_split'],
        scores['mean_test_score'],
        label=('Training accuracy'))

plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# Create the parameter grid
param_grid = {"max_depth": range(5, 15, 5),
             "min_samples_leaf": range(50, 150, 50),
             "min_samples_split": range(50, 150, 50),
             "criterion": ['gini', 'entropy']}

n_folds = 5

# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree,
                          param_grid = param_grid,
                          cv = n_folds,
                          verbose = 1)

# Fit the grid Search to the data
grid_search.fit(train_features, train_label)
# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.head(10)
# printing the optimal accuracy score and hyperparameters
print("best accuracy: ", grid_search.best_score_)
print(grid_search.best_estimator_)
# model with optimal hyperparameter
clf_gini = DecisionTreeClassifier(criterion='gini',
                                 random_state=100,
                                 max_depth=5,
                                 min_samples_leaf=50,
                                 min_samples_split=50)
clf_gini.fit(train_features, train_label)
# Plot the tree
data = export_graphviz(clf_gini, out_file=None, feature_names = features,
                       class_names = 'Survived', filled = True,
                      rounded = True, special_characters = True)
graph = graphviz.Source(data)
graph
test_df.isnull().sum()
test_df = pd.read_csv(test_path)
test_df.head()
meanFare = train_df['Fare'].mean()
test_df = test_df.fillna({"Fare": meanFare})

test_df = pd.get_dummies(test_df, columns=['Sex'])
test_df = pd.get_dummies(test_df, columns=['Embarked'])

ids = test_df['PassengerId']
test_feature = test_df[features]
predictions = clf_gini.predict(test_feature)

output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
output.head()
output.to_csv('submission.csv', index=False)
