import numpy as np
import pandas as pd 
!pip install skompiler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from skompiler import skompile
from sklearn import tree
#import and split the data  ( work on only one x variable)
diabetes = pd.read_csv("../input/diabetes/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df["Pregnancies"]
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
#set and fit the model
cart = DecisionTreeClassifier(max_depth=2)     # we input max_depth=2 for easy to understand the output
cart_model = cart.fit(X, y)
cart_model
#decision rule
!pip install skompiler
print(skompile(cart_model.predict).to("python/code"))   # we converted the rules to python code by skompile
x = [3]
((0 if x[0] <= 2.5 else 0) if x[0] <= 6.5 else 1 if x[0] <= 13.5 else 1)
#import and split the data  ( work on x variables)
diabetes = pd.read_csv("../input/diabetes/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
#set and fit the model
cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train, y_train)
cart_model
#decision rule
print(skompile(cart_model.predict).to("python/code"))   # we converted the rules to python code by skompile
#test error without tuning
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
#model tuning
#important params
#max_depth         : to control complexity
#min_samples_split : the min number of samples to split an internal node
#min_samples_leaf  : the min number of samples in last node

cart_params= {"max_depth": list(range(1,10)),
             "min_samples_split": list(range(2,50)) }
cart = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart, cart_params, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)
cart_cv_model.best_params_
cart_tuned_model= tree.DecisionTreeClassifier(max_depth= 5, min_samples_split= 19).fit(X_train, y_train)
#test error after tuning
y_pred=cart_tuned_model.predict(X_test)
accuracy_score(y_pred,y_test)
# We found 0.774 by Logistic Regression
#          0.775 by Naive Bayes 
#          0.731 by KNN
#          0.744 by Linear SVC
#          0.735 by Nonlinear SVC Steps
#          0.74  by ANN
#And now,  0.753 by CART