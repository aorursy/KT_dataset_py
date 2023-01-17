import pandas as pd 

# list for column headers

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']# open file with pd.read_csv

df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", names=names)

print(df.shape)# print head of data set

print(df.head())
#Defining indipendent and dependent variables

X = df.drop('class', axis=1)

y = df['class']
from sklearn.model_selection import train_test_split# implementing train-test-split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

rfc.fit(X_train, y_train)



# Predicting the Test set results

y_pred = rfc.predict(X_test)

# Importing cross validation for ?

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix
#cross validation 

rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
####### confusion matrix printing

print("=== Confusion Matrix ===")

print(confusion_matrix(y_test, y_pred))

print('\n')

print("=== Classification Report ===")

print(classification_report(y_test, y_pred))

print('\n')

print("=== All AUC Scores ===")

print(rfc_cv_score)

print('\n')

print("=== Mean AUC Score ===")

print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
import numpy as np

from sklearn.model_selection import RandomizedSearchCV # number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]# number of features at every split

max_features = ['auto', 'sqrt']



# max depth

max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]

max_depth.append(None)# create random grid

random_grid = {

 'n_estimators': n_estimators,

 'max_features': max_features,

 'max_depth': max_depth

 }# Random search of parameters

rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the model

rfc_random.fit(X_train, y_train)# print results

print(rfc_random.best_params_)
rfc = RandomForestClassifier(n_estimators=600, max_depth=300, max_features='sqrt')

rfc.fit(X_train,y_train)

rfc_predict = rfc.predict(X_test)

rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')

print("=== Confusion Matrix ===")

print(confusion_matrix(y_test, rfc_predict))

print('\n')

print("=== Classification Report ===")

print(classification_report(y_test, rfc_predict))

print('\n')

print("=== All AUC Scores ===")

print(rfc_cv_score)

print('\n')

print("=== Mean AUC Score ===")

print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())