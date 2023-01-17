#SVM on the Fisher's Iris Data set

#necessary libraries
import pandas as pd
import numpy as np
#ignore warnings
import warnings
warnings.filterwarnings('ignore')
#load the iris data set from sklearn's in-built datasets
from sklearn.datasets import load_iris
iris = load_iris()

iris.keys()
print(iris['target_names'])
#create a DataFrame for the features
iris_feat = pd.DataFrame(iris['data'],columns=iris['feature_names'])
iris_feat.head(2)
#train test split with random_state=101 so that we get the same results
from sklearn.model_selection import train_test_split
X = iris_feat
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
#importing the Support Vector Classifier, instantiating, fitting the model and predicting
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
pred = model.predict(X_test)
#see how the model performed
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#Wow! Notice that the model did pretty good! 
#Let's see if we can tune and the find the best parameters
#(unlikely, and we probably would be satisfied with these results)
#but just to practice GridSearch

from sklearn.model_selection import GridSearchCV

C_grid = [0.1,1,10,100,1000]
gamma_grid = [1,0.1,0.01,0.001,0.0001]
param_grid = {'C':C_grid, 'gamma':gamma_grid}
grid = GridSearchCV(SVC(),param_grid,verbose=2)
grid.fit(X_train,y_train)
#best parameter values
grid.best_params_
grid.best_estimator_
grid_pred = grid.predict(X_test)
print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))