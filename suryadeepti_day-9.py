import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv( 

'https://archive.ics.uci.edu/ml/machine-learning-'+

'databases/balance-scale/balance-scale.data', 

    sep= ',', header = None) 
len(data)
data.shape
data.head()
from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 
# Separating the target variable 

X = data.values[:, 1:5] 

Y = data.values[:, 0] 

  

# Splitting the dataset into train and test 

X_train, X_test, y_train, y_test = train_test_split(  

X, Y, test_size = 0.3, random_state = 100) 

#perform training with giniIndex.

clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5) 

  

    # Performing training 

clf_gini.fit(X_train, y_train) 
#perform training with entropy

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 3, min_samples_leaf = 5) 

  

    # Performing training 

clf_entropy.fit(X_train, y_train) 
y_pred = clf_gini.predict(X_test) 

print("Predicted values:") 

print(y_pred) 
print("Confusion Matrix: ", 

confusion_matrix(y_test, y_pred)) 

      

print ("Accuracy : ", 

accuracy_score(y_test,y_pred)*100) 

      

print("Report : ", 

classification_report(y_test, y_pred)) 
import matplotlib.pyplot as plt

plt.plot(y_test,y_pred)
plt.plot(y_test)
plt.plot(y_pred)