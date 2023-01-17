import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline
from sklearn import datasets

iris = datasets.load_iris()



X1_sepal = iris.data[:,[0,1]]

X2_petal = iris.data[:,[2,3]]

y = iris.target
plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.scatter(X1_sepal[:,0],X1_sepal[:,1],c=y)

plt.xlabel('Sepal length')

plt.ylabel('Sepal width')



plt.subplot(1,2,2)

plt.scatter(X2_petal[:,0],X2_petal[:,1],c=y)

plt.xlabel('Petal length')

plt.ylabel('Petal width')
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
from sklearn.cross_validation import train_test_split

# from sklearn.preprocessing import StandardScaler



#######################################################################

## SPLITTING





X_train_sepal, X_test_sepal, y_train_sepal, y_test_sepal = train_test_split(X1_sepal,y,test_size=0.3,random_state=0)



print("# training samples sepal: ", len(X_train_sepal))

print("# testing samples sepal: ", len(X_test_sepal))



X_train_petal, X_test_petal, y_train_petal, y_test_petal = train_test_split(X2_petal,y,test_size=0.3,random_state=0)



print("# training samples petal: ", len(X_train_petal))

print("# testing samples petal: ", len(X_test_petal))



#####################################################################

## SCALING ---> NOT REQUIRED!!



# sc = StandardScaler()

# X_train_sepal_std = sc.fit_transform(X_train_sepal)

# X_test_sepal_std = sc.transform(X_test_sepal)



# sc = StandardScaler()

# X_train_petal_std = sc.fit_transform(X_train_petal)

# X_test_petal_std = sc.transform(X_test_petal)
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.metrics import accuracy_score



tree = DecisionTreeClassifier()

tree.fit(X_train_sepal,y_train_sepal)



y_pred_sepal_tree_train = tree.predict(X_train_sepal)

y_pred_sepal_tree = tree.predict(X_test_sepal)



plot_decision_regions(X = X1_sepal

                      ,y = y

                      ,classifier = tree

                      ,test_idx = range(105,150))

plt.xlabel('Sepal length')

plt.ylabel('Sepal width')

plt.legend(loc='upper left')
tree = DecisionTreeClassifier()

tree.fit(X_train_petal,y_train_petal)



y_pred_petal_tree_train = tree.predict(X_train_petal)

y_pred_petal_tree = tree.predict(X_test_petal)



plot_decision_regions(X = X2_petal

                      ,y = y

                      ,classifier = tree

                      ,test_idx = range(105,150))

plt.xlabel('Petal length')

plt.ylabel('Petal width')

plt.legend(loc='upper left')
print("SEPAL")

print("------------------------------")

print("Training accuracy: %.2f" % accuracy_score(y_train_sepal,y_pred_sepal_tree_train))

print("Testing accuracy: %.2f" % accuracy_score(y_test_sepal,y_pred_sepal_tree))

print("")

print("PETAL")

print("------------------------------")

print("Training accuracy: %.2f" % accuracy_score(y_train_petal,y_pred_petal_tree_train))

print("Testing accuracy: %.2f" % accuracy_score(y_test_petal,y_pred_petal_tree))

plt.figure(figsize=(10, 10))



max_depth_range = [1,2,3,4,5,6]



j = 0

for i in max_depth_range:

    

    # Creating the decision tree

    tree = DecisionTreeClassifier(max_depth = i)

    tree.fit(X_train_sepal,y_train_sepal)

    

    # Printing decision regions

    plt.subplot(3,2,i)

    plt.subplots_adjust(hspace = 0.4)

    plot_decision_regions(X = X1_sepal

                      , y = y

                      , classifier = tree

                      , test_idx = range(105,150))

    plt.xlabel('Sepal length')

    plt.ylabel('Sepal width')

    plt.title('Max depth tree = %s'%i)
plt.figure(figsize=(10, 10))



max_depth_range = [1,2,3,4,5,6]



for i in max_depth_range:

    

    # Creating the decision tree

    tree = DecisionTreeClassifier(max_depth = i)

    tree.fit(X_train_petal,y_train_petal)

    

    # Printing decision regions

    plt.subplot(3,2,i)

    plt.subplots_adjust(hspace = 0.4)

    plot_decision_regions(X = X2_petal

                      , y = y

                      , classifier = tree

                      , test_idx = range(105,150))

    plt.xlabel('Petal length')

    plt.ylabel('Petal width')

    plt.title('Max depth tree = %s'%i)
from sklearn.learning_curve import validation_curve



max_depth_range = [1,2,3,4,5,6]



plt.figure(figsize=(15,10))



# Decision tree model

tree = DecisionTreeClassifier(max_depth=i)



# SEPAL validation curve

train_sepal_scores, test_sepal_scores = validation_curve(estimator = tree

                                                        , X = X1_sepal

                                                        , y = y

                                                        , param_name = 'max_depth'

                                                        , param_range = max_depth_range

                                                        , scoring = 'accuracy')



train_sepal_mean = np.mean(train_sepal_scores, axis = 1)

test_sepal_mean = np.mean(test_sepal_scores, axis = 1)



plt.subplot(2,2,1)

plt.plot(max_depth_range

        ,train_sepal_mean

        ,color='blue'

        ,marker='o'

        ,markersize=5

        ,label='training accuracy')



plt.plot(max_depth_range

        ,test_sepal_mean

        ,color='green'

        ,marker='x'

        ,markersize=5

        ,label='test accuracy') 



plt.legend(loc='upper left')

plt.xlabel('Max depth of tree')

plt.ylabel('Accuracy')

plt.title('Sepal validation curves')



# PETAL validation curve

train_petal_scores, test_petal_scores = validation_curve(estimator = tree

                                                        , X = X2_petal

                                                        , y = y

                                                        , param_name = 'max_depth'

                                                        , param_range = max_depth_range

                                                        , scoring = 'accuracy')



train_petal_mean = np.mean(train_petal_scores, axis = 1)

test_petal_mean = np.mean(test_petal_scores, axis = 1)



plt.subplot(2,2,2)

plt.plot(max_depth_range

        ,train_petal_mean

        ,color='blue'

        ,marker='o'

        ,markersize=5

        ,label='training accuracy')



plt.plot(max_depth_range

        ,test_petal_mean

        ,color='green'

        ,marker='x'

        ,markersize=5

        ,label='test accuracy') 



plt.legend(loc='lower right')

plt.xlabel('Max depth of tree')

plt.ylabel('Accuracy')

plt.title('Sepal validation curves')
from sklearn.ensemble import RandomForestClassifier



plt.figure(figsize=(10, 10))



max_depth_range = [1,2,3,4,5,6]



j = 0

for i in max_depth_range:

    

    # Creating the decision tree

    RF = RandomForestClassifier(max_depth = i)

    RF.fit(X_train_sepal,y_train_sepal)

    

    # Printing decision regions

    plt.subplot(3,2,i)

    plt.subplots_adjust(hspace = 0.4)

    plot_decision_regions(X = X1_sepal

                      , y = y

                      , classifier = RF

                      , test_idx = range(105,150))

    plt.xlabel('Sepal length')

    plt.ylabel('Sepal width')

    plt.title('Max depth tree = %s'%i)
plt.figure(figsize=(10, 10))



max_depth_range = [1,2,3,4,5,6]



j = 0

for i in max_depth_range:

    

    # Creating the decision tree

    RF = RandomForestClassifier(max_depth = i)

    RF.fit(X_train_petal,y_train_petal)

    

    # Printing decision regions

    plt.subplot(3,2,i)

    plt.subplots_adjust(hspace = 0.4)

    plot_decision_regions(X = X2_petal

                      , y = y

                      , classifier = RF

                      , test_idx = range(105,150))

    plt.xlabel('Petal length')

    plt.ylabel('Petal width')

    plt.title('Max depth tree = %s'%i)
max_depth_range = [1,2,3,4,5,6]



plt.figure(figsize=(15,10))



# Decision tree model

RF = RandomForestClassifier(max_depth = i)



# SEPAL validation curve

train_sepal_scores, test_sepal_scores = validation_curve(estimator = RF

                                                        , X = X1_sepal

                                                        , y = y

                                                        , param_name = 'max_depth'

                                                        , param_range = max_depth_range

                                                        , scoring = 'accuracy')



train_sepal_mean = np.mean(train_sepal_scores, axis = 1)

test_sepal_mean = np.mean(test_sepal_scores, axis = 1)



plt.subplot(2,2,1)

plt.plot(max_depth_range

        ,train_sepal_mean

        ,color='blue'

        ,marker='o'

        ,markersize=5

        ,label='training accuracy')



plt.plot(max_depth_range

        ,test_sepal_mean

        ,color='green'

        ,marker='x'

        ,markersize=5

        ,label='test accuracy') 



plt.legend(loc='upper left')

plt.xlabel('Max depth of tree')

plt.ylabel('Accuracy')

plt.title('Petal validation curves')



# PETAL validation curve

train_petal_scores, test_petal_scores = validation_curve(estimator = RF

                                                        , X = X2_petal

                                                        , y = y

                                                        , param_name = 'max_depth'

                                                        , param_range = max_depth_range

                                                        , scoring = 'accuracy')



train_petal_mean = np.mean(train_petal_scores, axis = 1)

test_petal_mean = np.mean(test_petal_scores, axis = 1)



plt.subplot(2,2,2)

plt.plot(max_depth_range

        ,train_petal_mean

        ,color='blue'

        ,marker='o'

        ,markersize=5

        ,label='training accuracy')



plt.plot(max_depth_range

        ,test_petal_mean

        ,color='green'

        ,marker='x'

        ,markersize=5

        ,label='test accuracy') 



plt.legend(loc='lower right')

plt.xlabel('Max depth of tree')

plt.ylabel('Accuracy')

plt.title('Petal validation curves')