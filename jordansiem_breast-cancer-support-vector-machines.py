#Breast Cancer Data
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from sklearn.datasets import load_breast_cancer
#Data coming from internal built - in set.
cancer = load_breast_cancer()
cancer.keys()
print(cancer['DESCR'])
#Is tumor malignant or benign? - All the values above are describing the tumor.
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()
cancer['target_names']
#Values above is difficult to interpret if you don't have industry knowledge.
from sklearn.model_selection import train_test_split
X = df_feat
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
#Predicted no tumors in predicted class or zero class
#Error....predicted everything to class one. Model needs parameters adjusted
#and data standardized.
#Find right parameters so we will use a gridsearch to find best
from sklearn.model_selection import GridSearchCV
#Dictionary of values to try. List of settings to be tested
#Keys we are using from the SVC - Look at model fit section.
#High C value gives you low bias and high variance. smaller c is opposite
#Gamma - free parame gaussian radio radial basis function
#also seen in kernel. Default is rbf.
#high gamla leads to high bias low var 
#So C and Gamma can change and model choose below
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,.01,.001,.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=3)

#Higher number the more verbose - text output. Don't leave as 0 default
grid.fit(X_train,y_train)
#What does this do?
#Fit is more involved. Run loop for cross validation and find best combo
#Then runs fit again on all data without cross validation
#to build new model with new settings. grab from grid object.
grid.best_params_
#Best cross valida score
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))

print(classification_report(y_test,grid_predictions))