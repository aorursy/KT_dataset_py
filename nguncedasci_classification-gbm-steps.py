import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
#import and split the data
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
gbm_model=GradientBoostingClassifier().fit(X_train,y_train)
#test error without tuning
y_pred=gbm_model.predict(X_test)
accuracy_score(y_pred,y_test)
#model tuning
gbm_model
#important parameters
#learning_rate
#n_estimators
#subsample : if the data set is too large, it can be used with other values
#min_samples_split
#min_samples_leaf
#max_depth
gbm_params=gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],
             "n_estimators": [100,500,100],
             "max_depth": [3,5,10],
             "min_samples_split": [2,5,10]}
gbm = GradientBoostingClassifier()
gbm_cv = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)
gbm_cv.fit(X_train, y_train)
gbm_cv.best_params_
gbm_tuned_model= GradientBoostingClassifier(learning_rate= 0.05,
 max_depth=5,
 min_samples_split=10,
 n_estimators=100).fit(X_train,y_train)
y_pred=gbm_tuned_model.predict(X_test)
accuracy_score(y_test,y_pred)
# We found 0.774 by Logistic Regression
#          0.775 by Naive Bayes 
#          0.731 by KNN
#          0.744 by Linear SVC
#          0.735 by Nonlinear SVC Steps
#          0.74  by ANN
#          0.753 by CART
#          0.735 by Random Forest
#And now,  0.735 by GBM