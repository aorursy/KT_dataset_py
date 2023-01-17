import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from catboost import CatBoostClassifier
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
catb_model=CatBoostClassifier().fit(X_train,y_train)
#test error without tuning
y_pred=catb_model.predict(X_test)
accuracy_score(y_pred,y_test)
#model tuning
#important parameters
catb_params = {
    'iterations': [200,500],
    'learning_rate': [0.01,0.05],
    'depth': [3,5] }
catb_cv_model= GridSearchCV(catb_model,catb_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
catb_cv_model.best_params_
catb_tuned_model= CatBoostClassifier(iterations = 200, 
                          learning_rate = 0.01, 
                          depth = 8).fit(X_train, y_train)
y_pred=catb_tuned_model.predict(X_test)

accuracy_score(y_test,y_pred)
# We found 0.766 by Logistic Regression
#          0.775 by Naive Bayes 
#          0.731 by KNN
#          0.744 by Linear SVC
#          0.735 by Nonlinear SVC Steps
#          0.735  by ANN
#          0.753 by CART
#          0.735 by Random Forests
#          0.735 by GBM
#          0.757 by XG Boost
#          0.744 by Light GBM
#And now,  0.753 by CatBoost
#Naive Bayes is the best model for this dataset.
