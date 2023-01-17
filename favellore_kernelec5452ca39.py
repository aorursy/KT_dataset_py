# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np



import matplotlib.pyplot as plt



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn import datasets



iris = datasets.load_iris()

X = iris.data[:,[2,3]]

Y = iris.target
plt.figure(2, figsize=(8, 6))

plt.clf() #clear figure



# Plot the training points

plt.scatter(X[:, 0], X[:, 1], c=Y)

plt.xlabel('Petal length')

plt.ylabel('Petal width')
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)



length_Train = len(X_train)

length_Test = len(X_test)



print("There are ",length_Train,"samples in the trainig set and",length_Test,"samples in the test set")

print("-----------------------------------------------------------------------------------------------")

print("")



## 2. Feature scaling.



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_standard = sc.transform(X_train)

X_test_standard = sc.transform(X_test)



print("X_train without standardising features")

print("--------------------------------------")

print(X_train[1:5,:])

print("")

print("X_train standardising features")

print("--------------------------------------")

print(X_train_standard[1:5,:])
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C = 1000.0, random_state = 0 )

lr.fit(X_train_standard, Y_train)
Y_pred_Logit = lr.predict(X_test_standard)

from sklearn.metrics import accuracy_score

print("Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_Logit))
from matplotlib.colors import ListedColormap



def plot_decision_regions(X,y,classifier,test_idx=None,resolution=0.02):

    

    # Initialise the marker types and colors

    markers = ('s','x','o','^','v')

    colors = ('red','blue','lightgreen','gray','cyan')

    color_Map = ListedColormap(colors[:len(np.unique(y))]) #we take the color mapping correspoding to the 

                                                            #amount of classes in the target data

    

    # Parameters for the graph and decision surface

    x1_min = X[:,0].min() - 1

    x1_max = X[:,0].max() + 1

    x2_min = X[:,1].min() - 1

    x2_max = X[:,1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),

                           np.arange(x2_min,x2_max,resolution))

    

    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    

    plt.contour(xx1,xx2,Z,alpha=0.4,cmap = color_Map)

    plt.xlim(xx1.min(),xx1.max())

    plt.ylim(xx2.min(),xx2.max())

    

    # Plot samples

    X_test, Y_test = X[test_idx,:], y[test_idx]

    

    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],

                    alpha = 0.8, c = color_Map(idx),

                    marker = markers[idx], label = cl

                   )

X_combined_standard = np.vstack((X_train_standard,X_test_standard))

Y_combined = np.hstack((Y_train, Y_test))

plot_decision_regions(X = X_combined_standard

                      , y = Y_combined

                      , classifier = lr

                      , test_idx = range(105,150))
from sklearn.svm import SVC



svm = SVC(kernel = 'linear', C = 1.0, random_state = 0)

svm.fit(X_train_standard, Y_train)
Y_pred_SVM = svm.predict(X_test_standard)



print("Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_SVM))
plot_decision_regions(X = X_combined_standard

                      , y = Y_combined

                      , classifier = svm

                      , test_idx = range(105,150))
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(criterion='entropy'

                             , max_depth = 3

                             , random_state = 0)

tree.fit(X_train,Y_train)
Y_pred_tree = tree.predict(X_test)



print("Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_tree))
X_combined = np.vstack((X_train, X_test))

Y_combined = np.hstack((Y_train, Y_test))



plot_decision_regions(X = X_combined

                      , y = Y_combined

                      , classifier = tree

                      , test_idx = range(105,150))
from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(criterion = 'entropy'

                                , n_estimators = 10

                                , random_state = 1

                                , n_jobs = 1)



forest.fit(X_train, Y_train)
Y_pred_RF = forest.predict(X_test)



print("Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_RF))
plot_decision_regions(X = X_combined

                      , y = Y_combined

                      , classifier = forest

                      , test_idx = range(105,150))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 5

                          , p=2

                          , metric = 'minkowski')





knn.fit(X_train_standard,Y_train)
Y_pred_KNN = knn.predict(X_test_standard)



print("Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_KNN))
plot_decision_regions(X = X_combined_standard

                      , y = Y_combined

                      , classifier = knn

                      , test_idx = range(105,150))
plt.figure(figsize=(10, 10))







plt.subplot(3,2,2)

plot_decision_regions(X = X_combined_standard

                      , y = Y_combined

                      , classifier = lr

                      , test_idx = range(105,150))



plt.subplot(3,2,3)

plot_decision_regions(X = X_combined_standard

                      , y = Y_combined

                      , classifier = svm

                      , test_idx = range(105,150))



plt.subplot(3,2,4)

plot_decision_regions(X = X_combined

                      , y = Y_combined

                      , classifier = tree

                      , test_idx = range(105,150))



plt.subplot(3,2,5)

plot_decision_regions(X = X_combined

                      , y = Y_combined

                      , classifier = forest

                      , test_idx = range(105,150))



plt.subplot(3,2,6)

plot_decision_regions(X = X_combined_standard

                      , y = Y_combined

                      , classifier = knn

                      , test_idx = range(105,150))