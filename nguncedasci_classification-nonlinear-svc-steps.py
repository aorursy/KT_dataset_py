# RBF SVC STEPS
#related libraries
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#import and split the data
diabetes = pd.read_csv("../input/diabetes/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
#set and fit the model
svc_model=SVC(kernel="rbf",gamma='auto').fit(X_train,y_train)   # Actually,we don't have to indicate "rbf" 
                                                   #because rbf is default setting for SVC kernel. 
svc_model
y_pred= svc_model.predict(X_test)
accuracy_score(y_test,y_pred)
#model tuning
# we will tune "c value " and "gamma" which are hiperparameters for this model
svc_params={"C": [0.0001,0.001,0.01,0.1,1,5,10,50,100],"gamma":[0.0001,0.001,0.01,0.1,1,5,10,50,100]}
svc_cv_model= GridSearchCV(svc_model,svc_params,cv=10,n_jobs=-1,verbose=2)
svc_cv_model.fit(X_train,y_train)
svc_cv_model.best_params_
svc_final_model=SVC(kernel="rbf", C=10,gamma=0.0001).fit(X_train,y_train)
y_pred=svc_final_model.predict(X_test)
accuracy_score(y_test,y_pred)
# We found 0.766 by Logistic Regression
#          0.775 by Naive Bayes 
#          0.731 by KNN
#          0.744 by Linear SVC
#And now,  0.735 by Nonlinear SVC Steps
