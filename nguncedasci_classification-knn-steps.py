# KNN Steps
#import and split
#set and fit
#prediction
#model tuning(n_neighbors)
#set and fit the final model
#test error

#related libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#import, split
diabetes=pd.read_csv("../input/diabetes/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
#set and fit the model
knn_model= KNeighborsClassifier().fit(X_train,y_train)
knn_model
#prediction
y_pred= knn_model.predict(X_test)
#accuracy score (not verified)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
#model tuning: n_neighbors is our hiperparameter in here.
knn_params={ "n_neighbors": np.arange(1,50)}
knn_cv= GridSearchCV(knn_model, knn_params,cv=10)
knn_cv.fit(X_train,y_train)
print("The best params are:", knn_cv.best_params_)
#final model
knn_tuned=KNeighborsClassifier(n_neighbors= 11).fit(X_train,y_train)
knn_tuned.score(X_test,y_test)
# or
y_pred=knn_tuned.predict(X_test)
accuracy_score(y_test,y_pred)
# We found 0.766 by Logistic Regression
#          0.775 by Naive Bayes 
#And now,  0.731 by KNN
#Thanks to https://github.com/mvahit/DSMLBC
