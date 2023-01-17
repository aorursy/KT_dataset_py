# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten
data = pd.read_csv("../input/Absenteeism_at_work.csv")
#1st column to 13th column
X =  data.iloc[:,range(0,13)]
#14st column --> Labels
Y =  data.iloc[:,14]

#normalising the data
normal_X = preprocessing.normalize(X)
#Splitting of data set
Train_X, Test_X, Train_y, Test_y = train_test_split(X, Y, test_size = 0.2, random_state = 239)
Train_y = Train_y.ravel()
Test_y = Test_y.ravel()
k = []
accuracy = []
for i in range(int(np.sqrt(data.shape[0]))):
    K_value = i+1
    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
    neigh.fit(Train_X, Train_y) 
    Pred_y = neigh.predict(Test_X)
    #print(Pred_y)
    k.append(K_value)
    accuracy.append(accuracy_score(Test_y,Pred_y)*100)
    
print("Accuracy of the model: ",max(accuracy))
print("K value: ",accuracy.index(max(accuracy))+1)
#graph for visualization
plt.plot(k, accuracy)
plt.xlabel('K Values')
plt.ylabel('Accuracy')
plt.title('K values vs Accuracy graph')
plt.show()
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(Train_X, Train_y)
Pred_y_gini = clf_gini.predict(Test_X)
print(accuracy_score(Test_y,Pred_y_gini)*100)
print(confusion_matrix(Test_y,Pred_y_gini))
print(classification_report(Test_y,Pred_y_gini))
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(Train_X, Train_y)
Pred_y_entropy = clf_entropy.predict(Test_X)
print(accuracy_score(Test_y,Pred_y_entropy)*100)
print(confusion_matrix(Test_y,Pred_y_entropy))
print(classification_report(Test_y,Pred_y_entropy))

