import pandas as pd

import numpy as np



import matplotlib.pyplot as plt



%matplotlib inline

from sklearn import datasets

iris = datasets.load_iris()



X = iris.data[:,[2,3]]

Y = iris.target



print(X[1:5,:])

print(Y)                      #check that instead of having the names of the type of flower, these have been enccoded
plt.figure(2, figsize=(8, 6))

plt.clf() #clear figure



# Plot the training points

plt.scatter(X[:, 0], X[:, 1], c=Y)

plt.xlabel('Petal length')

plt.ylabel('Petal width')

## 1. Splitting the dataset



from sklearn.cross_validation import train_test_split

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
example_train_data = [10,20,30]

example_sc = StandardScaler()                                       #calculated sample mean and standard deviation

example_sc.fit(example_train_data)

example_train_data_scaled = example_sc.transform(example_train_data)   



print(example_train_data)

print(example_train_data_scaled)



print("----------------------------------------------")

print("")

print("What would happen if, instead of scaling the test dataset with the training scaling parameters, we scaled")

print("with the test scaling parameters?")



example_test_data = [5,6,7]

example_sc = StandardScaler()

example_sc.fit(example_test_data)

example_test_data_scaled = example_sc.transform(example_test_data)



print(example_test_data)

print(example_test_data_scaled)



print("")

print("If you observe, the test dataset, which has much smaller values than the train dataset, has the same scaled")

print("values!! Intuitively, being 5,6,7 (test set) much lower than 10,20,30 (train set), they should be weighted ")

print("to show values much lower than -1.22  which represents the weighted value for 10 in the train dataset.")

print("----------------------------------------------")

print("")

print("Let's apply the correct scaling method")



example_sc = StandardScaler()

example_sc.fit(example_train_data)

example_test_data_scaled = example_sc.transform(example_test_data)



print(example_test_data)

print(example_test_data_scaled)

from sklearn.linear_model import Perceptron



## Initialise the perceptron model with:

##   Max number of iterations = 40

##   Learning rate = 0.1

ppn = Perceptron(n_iter = 40, eta0 = 0.1, random_state = 0)

ppn.fit(X_train_standard, Y_train)
Y_pred_perceptron = ppn.predict(X_test_standard)



from sklearn.metrics import accuracy_score

print("Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_perceptron))
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

                      , classifier = ppn

                      , test_idx = range(105,150))



print("")

print("If you are familiar with the perceptron model, you will recognise that the algorithn will never converge"

     "(perfectly classify) datasets that are not linearly separable. This is why the perceptron model is not recommend"

     "for being to simple and weak")
# X_train, X_test, Y_train, Y_test

# X_train_standard, X_test_standard



from sklearn.linear_model import LogisticRegression



## Initialise the logistc regression model with:

##   C (regularization parameter) = 1000

lr = LogisticRegression(C = 1000.0, random_state = 0 )

lr.fit(X_train_standard, Y_train)
Y_pred_Logit = lr.predict(X_test_standard)



print("Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_Logit))
plot_decision_regions(X = X_combined_standard

                      , y = Y_combined

                      , classifier = lr

                      , test_idx = range(105,150))
# X_train, X_test, Y_train, Y_test

# X_train_standard, X_test_standard



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
print("Perceptron Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_perceptron))

print("Logistic Regression Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_Logit))

print("SVM Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_SVM))

print("Decision Tree Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_tree))

print("Random Forest Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_RF))

print("KNN Accuracy: %.2f" % accuracy_score(Y_test,Y_pred_KNN))
# 7.2. SUMMARY: Plots
plt.figure(figsize=(10, 10))



plt.subplot(3,2,1)

plot_decision_regions(X = X_combined_standard

                      , y = Y_combined

                      , classifier = ppn

                      , test_idx = range(105,150))



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