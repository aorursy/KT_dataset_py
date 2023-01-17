# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sn

import matplotlib.pyplot as plt # for visualization purposes

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Exploring first five elements in the entire dataset and define useful variables

dataset_train = pd.read_csv("../input/train.csv")

m = dataset_train.shape[0]

y = dataset_train['Survived']

dataset_train = dataset_train.drop('Survived', 1)

dataset_test = pd.read_csv("../input/test.csv")

dataset = pd.concat([dataset_train, dataset_test], axis=0)

dataset.head()
# Dataset' stats

dataset.describe()
# Correlation between numerical variables

print(dataset.corr())

plt.matshow(dataset.corr()) # excluding categorical variables
# we can see how 'Pclass' and 'Age' variables are strongly correlated.

# Likewise, 'Pclass' and 'Fare' are too correlated. Better the class (class1 > class3) higher price.
# A better look to the 'Cabin' variable

#dataset['Cabin']
# A better look to the 'Ticket' variable

#dataset['Ticket']
# Split dataset into numerical columns, categorical columns and objetive column 'survived'.

numerical_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

categorical_columns = ['Sex', 'Embarked']

dataset_numerical = dataset[numerical_columns]

dataset_categorical = dataset[categorical_columns]

dataset_numerical = dataset_numerical.fillna(0.0) # filling null values with 0.0

sn.pairplot(dataset_numerical)
# Preprocessing numerical columns

dataset_numerical[['Age', 'Fare']] -= dataset_numerical[['Age', 'Fare']].min()

dataset_numerical[['Age', 'Fare']] /= dataset_numerical[['Age', 'Fare']].max()

dataset_numerical = pd.get_dummies(dataset_numerical, columns=['Pclass'])

dataset_numerical.head()
# Preprocessing categorical columns

dataset_categorical = pd.get_dummies(dataset_categorical)

dataset_categorical.head()
concat = pd.concat([dataset_numerical, dataset_categorical], axis=1)

X = concat.iloc[:m, :]

X_for_test_predictions = concat.iloc[m:, :]

concat.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ids = dataset['PassengerId'].iloc[m:] # ids dor predictions

def save_predictions(predictions, path):

    result = pd.DataFrame(predictions, index=ids, columns=['Survived'])

    result.to_csv(path)
# Training phase

from sklearn.model_selection import GridSearchCV # include cross-validation

# Logistic regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

grid = {'penalty': ['l1', 'l2'], 'C':[0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2]}

model_lr = GridSearchCV(lr, grid, scoring='accuracy')

model_lr.fit(X_train, y_train)

print('Logistic regression best score: ', model_lr.best_score_)

# random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

grid = {'n_estimators': [x*10 for x in range(1, 10)], 'criterion': ['gini', 'entropy']}

model_rf = GridSearchCV(rf, grid, scoring='accuracy')

model_rf.fit(X_train, y_train)

print('Random forest best score: ', model_rf.best_score_)

# SVC

from sklearn.svm import SVC

svc = SVC()

grid = {'C': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 

        'degree': [2, 3, 4]}

model_svc = GridSearchCV(svc, grid, scoring='accuracy')

model_svc.fit(X_train, y_train)

print('Support vector best score: ', model_svc.best_score_)
# Test results

# Logistic regression

y_pred_lr = model_lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_lr)

print("Accuracy logistic regression: ", accuracy)

predictions_logistic_regression = model_lr.predict(X_for_test_predictions)

# Random forest

y_pred_rf = model_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_rf)

print("Accuracy random forest: ", accuracy)

predictions_random_forest = model_rf.predict(X_for_test_predictions)

# Support vector machine

y_pred_svc = model_svc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_svc)

print("Accuracy support vector machine: ", accuracy)

predictions_svc = model_svc.predict(X_for_test_predictions)
# Print best estimator

print(model_lr.best_estimator_)

print(model_rf.best_estimator_)

print(model_svc.best_estimator_)
#save_predictions(predictions_random_forest, 'predictions_random_forest.csv')
# Compute metrics

import seaborn as sn

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score



cm_lr = confusion_matrix(y_test, y_pred_lr)

sn.heatmap(cm_lr, annot=True)

precision_lr = precision_score(y_test, y_pred_lr)

recall_lr = recall_score(y_test, y_pred_lr)

F1_lr = 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr)

print('precision_lr:', precision_lr)

print('recall_lr:', recall_lr)

print('F1 score:', F1_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)

sn.heatmap(cm_rf, annot=True)

precision_rf = precision_score(y_test, y_pred_rf)

recall_rf = recall_score(y_test, y_pred_rf)

F1_rf = 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

print('precision_rf:', precision_rf)

print('recall_rf:', recall_rf)

print('F1 score:', F1_rf)
cm_svc = confusion_matrix(y_test, y_pred_svc)

sn.heatmap(cm_svc, annot=True)

precision_svc = precision_score(y_test, y_pred_svc)

recall_svc = recall_score(y_test, y_pred_svc)

F1_svc = 2 * (precision_svc * recall_svc) / (precision_svc + recall_svc)

print('precision_svc:', precision_svc)

print('recall_svc:', recall_svc)

print('F1 score:', F1_svc)