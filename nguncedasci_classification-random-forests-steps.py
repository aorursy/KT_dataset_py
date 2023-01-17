import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
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
rf_model=RandomForestClassifier().fit(X_train,y_train)
#test error without tuning
y_pred=rf_model.predict(X_test)
accuracy_score(y_pred,y_test)
rf_model
#model tuning
#very important parameter:
#max_features: the number of features to consider while splitting

#important parameters
#n_estimators: number of trees. 500 and 1000 are better than default setting(10) according to some scientific studies depend on data
#max_depth  : 
#min_samples_split

rf_params={"max_features":[2,5,8], "max_depth":[2,3,5,8,10],"n_estimators":[10,500,1000], "min_samples_split":[2,5,10] }
rf_tuned_model=GridSearchCV(rf_model, rf_params, cv=10,n_jobs=-1, verbose=2).fit(X_train,y_train)
rf_tuned_model.best_params_
rf_tuned_model=RandomForestClassifier(max_depth=8, max_features=8, min_samples_split=2, n_estimators=10).fit(X_train,y_train)
#test error after tuning
y_pred=rf_tuned_model.predict(X_test)
accuracy_score(y_test,y_pred)
# We found 0.774 by Logistic Regression
#          0.775 by Naive Bayes 
#          0.731 by KNN
#          0.744 by Linear SVC
#          0.735 by Nonlinear SVC Steps
#          0.74  by ANN
#          0.753 by CART
#And now,  0.735 by Random Forest
#Importance Level of Features 
Importance = pd.DataFrame({"Importance": rf_tuned_model.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Importance Level of Features");






