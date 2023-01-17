import pandas as pd

import numpy

from sklearn.datasets import load_iris
iris=load_iris()
X = iris.data 

y = iris.target 

  

# splitting X and y into training and testing sets 

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1) 
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors=3) 

knn.fit(X_train, y_train) 
y_pred = knn.predict(X_test) 
from sklearn import metrics 

print("kNN model accuracy:", metrics.accuracy_score(y_test, y_pred)) 
sample = [[3, 5, 4, 2], [2, 3, 5, 4]] 

preds = knn.predict(sample) 

pred_species = [iris.target_names[p] for p in preds] 

print("Predictions:", pred_species) 

  
from sklearn.externals import joblib 

joblib.dump(knn, 'iris_knn.pkl')