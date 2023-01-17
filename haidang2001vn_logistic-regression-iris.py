import numpy as np

from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import svm
#Load Iris dataset

iris = load_iris()

# print(iris.DESCR)

X_iris = iris.data

y_iris = iris.target

print("First sample:", X_iris[0], y_iris[0])

print(y_iris)

#Check the shape of data

print (X_iris.shape)



#Check if sets balanced

print ('Data: {}, 2: {}, 3: {}'.format(np.sum(y_iris == 0), np.sum(y_iris == 1), np.sum(y_iris == 2) ) )



# X_twoFeature = iris.data[:, [0,1]]  # we only take the first two features.

X_twoFeature = iris.data[:, [0,1,2]] 

y = iris.target

markers = ('s', 'x', 'o')

colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

cmap = ListedColormap(colors[:len(np.unique(y))])

plt.scatter(x=X_twoFeature[y == 0, 0], y=X_twoFeature[y == 0, 1], c=cmap(0), marker=markers[0], label=[iris.target_names[0]])

plt.scatter(x=X_twoFeature[y == 1, 0], y=X_twoFeature[y == 1, 1], c=cmap(1), marker=markers[1], label=[iris.target_names[1]])

plt.scatter(x=X_twoFeature[y == 2, 0], y=X_twoFeature[y == 2, 1], c=cmap(2), marker=markers[2], label=[iris.target_names[2]])

plt.legend()
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):



    # setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])



    # plot the decision surface

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),

                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())



    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],

                    alpha=0.8, c=cmap(idx),

                    marker=markers[idx], label=cl)

    

    # highlight test samples

    if test_idx:

        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1], alpha=1.0, c='yellow', linewidth=1, marker='>',

                   s=10, label="test set")
X_train, X_test, y_train, y_test = train_test_split(X_twoFeature, y, test_size=0.3, random_state=0)

#Create estimator class

# model_two = LogisticRegression(solver='newton-cg', multi_class='multinomial')

print(model_two)

model_two = svm.SVC(kernel='poly')

model_two.fit(X_train, y_train)

# parameters = model_two.coef_

predicted_classes = model_two.predict(X_test)

accuracy = accuracy_score(predicted_classes,y_test)

print('The accuracy score using scikit-learn is {}'.format(accuracy))

print("The model parameters using scikit learn")

print(parameters)

print("confusion_matrix")

print(confusion_matrix(predicted_classes,y_test))



X_combined= np.vstack((X_train, X_test))

y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined, y=y_combined, classifier=model_two, test_idx=range(105,150,150))

plt.legend(loc='upper left')

plt.show()
#Make split for original data

(X_tr_o, X_ts_o, y_tr_o, y_ts_o ) = train_test_split(X_iris, y_iris, stratify=y_iris, test_size= 0.3, random_state=0)

print(X_tr_o.shape)

print(y_tr_o.shape)

print(X_ts_o.shape)

print(y_ts_o.shape)
#Create estimator class

model = svm.SVC(kernel='rbf')

# model = LogisticRegression(C = 5, penalty='l2')

print(model)

model.fit(X_tr_o, y_tr_o)

# parameters = model.coef_

predicted_classes = model.predict(X_ts_o)

accuracy = accuracy_score(predicted_classes,y_ts_o)

print('The accuracy score using scikit-learn is {}'.format(accuracy))

print("The model parameters using scikit learn")

print(parameters)

print("confusion_matrix")

print(confusion_matrix(predicted_classes,y_ts_o))