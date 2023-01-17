import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from xgboost import XGBClassifier
!pip install xgboost
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
xgb_model=XGBClassifier().fit(X_train,y_train)
#test error without tuning
y_pred=xgb_model.predict(X_test)
accuracy_score(y_test,y_pred)
#model tuning
#important parameters
#max_depth
#learning_rate
#n_estimators (number of iterations)
#booster: gbtree
#subsample
#colsample_bytree
xgb_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'learning_rate': [0.1,0.01,0.02,0.05],
        "min_samples_split": [2,5,10]}
xgb_cv=GridSearchCV(xgb_model,xgb_params, cv=10, n_jobs=-1, verbose=2)
xgb_cv.fit(X_train,y_train)
xgb_cv.best_params_
xgb_tuned=XGBClassifier(learning_rate=0.02, max_depth=3, n_estimators=100, subsample=0.6).fit(X_train,y_train)
y_pred=xgb_tuned.predict(X_test)
accuracy_score(y_test,y_pred)
# We found 0.766 by Logistic Regression
#          0.775 by Naive Bayes 
#          0.731 by KNN
#          0.744 by Linear SVC
#          0.735 by Nonlinear SVC Steps
#          0.735 by ANN
#          0.753 by CART
#          0.735 by Random Forest
#          0.735 by GBM
#And now,  0.757 by XG Boost
