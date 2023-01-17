!conda install -c conda-forge lightgbm
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from lightgbm import LGBMClassifier
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
lgbm_model=LGBMClassifier().fit(X_train,y_train)
#test error without tuning
y_pred=lgbm_model.predict(X_test)
accuracy_score(y_pred,y_test)
#model tuning
#important parameters
#n_estimators
#subsample
#max_depth
#learning_rate
#min_child_samples  : minimum number of data needed in leaf node
lgbm_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'learning_rate': [0.1,0.01,0.02,0.05],
        "min_child_samples": [5,10,20]}
lgbm_cv= GridSearchCV(lgbm_model,lgbm_params, cv=10,n_jobs=-1, verbose=2)
lgbm_cv.fit(X_train, y_train)
lgbm_cv.best_params_
lgbm_tuned_model= LGBMClassifier( learning_rate= 0.01,
 max_depth= 3,
 min_child_samples=20,
 n_estimators=500,
 subsample=0.6). fit(X_train,y_train)
y_pred=lgbm_tuned_model.predict(X_test)
accuracy_score(y_pred,y_test)
# We found 0.766 by Logistic Regression
#          0.775 by Naive Bayes 
#          0.731 by KNN
#          0.744 by Linear SVC
#          0.735 by Nonlinear SVC Steps
#          0.735  by ANN
#          0.753 by CART
#          0.735 by Random Forest
#          0.735 by GBM
#          0.757 by XG Boost
#And now,  0.744 by Light GBM
