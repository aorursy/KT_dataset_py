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
import matplotlib.pyplot as plt
df=pd.read_csv('../input/diabetes.csv')
df.head()
X=df.drop('Outcome',axis=1).values

y=df.Outcome.values

X=X.reshape(-1,8)

y=y.reshape(-1,)
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)



# Instantiate a k-NN classifier: knn

knn = KNeighborsClassifier(n_neighbors=6)



# Fit the classifier to the training data

knn.fit(X_train,y_train)



# Predict the labels of the test data: y_pred

y_pred = knn.predict(X_test)



# Generate the confusion matrix and classification report

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='lbfgs')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)



# Create the classifier: logreg

logreg = LogisticRegression(solver='lbfgs')



# Fit the classifier to the training data

logreg.fit(X_train,y_train)



# Compute predicted probabilities: y_pred_prob

y_pred_prob = logreg.predict_proba(X_test)[:,1]



# Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
# Import necessary modules

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score



# Compute predicted probabilities: y_pred_prob

y_pred_prob = logreg.predict_proba(X_test)[:,1]



# Compute and print AUC score

print("AUC: {}".format(roc_auc_score(y_test,y_pred_prob)))



# Compute cross-validated AUC scores: cv_auc

cv_auc = cross_val_score(logreg,X,y,cv=5, scoring='roc_auc')



# Print list of AUC scores

print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

# Import necessary modules

from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import GridSearchCV



# Setup the hyperparameter grid

c_space = np.logspace(-5, 8, 15)

param_grid = {'C': c_space}



# Instantiate a logistic regression classifier: logreg

logreg = LogisticRegression(solver='liblinear')



# Instantiate the GridSearchCV object: logreg_cv

logreg_cv = GridSearchCV(logreg, param_grid, cv=5)



# Fit it to the data

logreg_cv.fit(X,y)



# Print the tuned parameters and score

print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 

print("Best score is {}".format(logreg_cv.best_score_))

# Import necessary modules

from scipy.stats import randint

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RandomizedSearchCV



# Setup the parameters and distributions to sample from: param_dist

param_dist = {"max_depth": [3, None],

              "max_features": randint(1, 9),

              "min_samples_leaf": randint(1, 9),

              "criterion": ["gini", "entropy"]}



# Instantiate a Decision Tree classifier: tree

tree = DecisionTreeClassifier()



# Instantiate the RandomizedSearchCV object: tree_cv

tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)



# Fit it to the data

tree_cv.fit(X,y)



# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))

print("Best score is {}".format(tree_cv.best_score_))

# Import necessary modules

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



# Create the hyperparameter grid

c_space = np.logspace(-5, 8, 15)

param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}



# Instantiate the logistic regression classifier: logreg

logreg = LogisticRegression(solver='liblinear')



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)



# Instantiate the GridSearchCV object: logreg_cv

logreg_cv = GridSearchCV(logreg,param_grid,cv=5)



# Fit it to the training data

logreg_cv.fit(X_train,y_train)



# Print the optimal parameters and best score

print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))

print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

# Import necessary modules

from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV, train_test_split



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)



# Create the hyperparameter grid

l1_space = np.linspace(0, 1, 30)

param_grid = {'l1_ratio': l1_space}



# Instantiate the ElasticNet regressor: elastic_net

elastic_net = ElasticNet()



# Setup the GridSearchCV object: gm_cv

gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)



# Fit it to the training data

gm_cv.fit(X_train,y_train)



# Predict on the test set and compute metrics

y_pred = gm_cv.predict(X_test)

r2 = gm_cv.score(X_test, y_test)

mse = mean_squared_error(y_test, y_pred)

print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))

print("Tuned ElasticNet R squared: {}".format(r2))

print("Tuned ElasticNet MSE: {}".format(mse))

df.head()
df.Insulin.replace(0,np.nan,inplace=True)

df.SkinThickness.replace(0,np.nan,inplace=True)

df.BMI.replace(0,np.nan,inplace=True)

df.info()
df.drop