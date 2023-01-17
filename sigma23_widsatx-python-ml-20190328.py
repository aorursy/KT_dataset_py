from IPython.display import Image
Image("../input/images/images/what_is_ml.png")
Image("../input/images/images/types_oh_ml.png")
Image("../input/images/images/types_of_ml.png")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import sklearn



# Any results you write to the current directory are saved as output.
!pip install watermark
%load_ext watermark
%watermark --iversions
Image("../input/images/images/bias_variance_tradeoff_reg.png")
Image("../input/images/images/overfitting_under_classification.png")
from sklearn import datasets

import numpy as np



iris = datasets.load_iris() # loading built in data set

X = iris.data[:, [2, 3]] # Here, the third column represents the petal length, 

                        # and the fourth column the petal width of the flower samples.

y = iris.target # The classes are already converted to integer labels 

                # .where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.



print('Class labels:', np.unique(y))
X.shape # shape of the matrix, # of samples by # of features
plt.scatter(X[:,0], X[:,1])

plt.xlabel('petal length')

plt.ylabel('petal width')

plt.show()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=1, stratify=y)
print('Labels counts in y:', np.bincount(y))

print('Labels counts in y_train:', np.bincount(y_train))

print('Labels counts in y_test:', np.bincount(y_test))
plt.scatter(X_train[:,0], X_train[:,1])

plt.xlabel('petal length')

plt.ylabel('petal width')
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))

y_combined = np.hstack((y_train, y_test))
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt





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

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())



    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], 

                    y=X[y == cl, 1],

                    alpha=0.8, 

                    c=colors[idx],

                    marker=markers[idx], 

                    label=cl, 

                    edgecolor='black')



    # highlight test samples

    if test_idx:

        # plot all samples

        X_test, y_test = X[test_idx, :], y[test_idx]



        plt.scatter(X_test[:, 0],

                    X_test[:, 1],

                    c='',

                    edgecolor='black',

                    alpha=1.0,

                    linewidth=1,

                    marker='o',

                    s=100, 

                    label='test set')
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=20, 

                           p=2, 

                           metric='minkowski') #euclidean distance

knn.fit(X_train_std, y_train)



plot_decision_regions(X_combined_std, y_combined, 

                      classifier=knn, test_idx=range(105, 150))



plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.tight_layout()

#plt.savefig('images/03_24.png', dpi=300)

plt.show()
y_pred = knn.predict(X_test_std)
knn.score(X_train_std, y_train)
knn.score(X_test_std, y_test) # 98% accurate
from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
np.set_printoptions(precision=2)

class_names = iris.target_names

# Plot non-normalized confusion matrix

plot_confusion_matrix(y_test, y_pred, classes=class_names,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')



plt.show()
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(criterion='gini', 

                              max_depth=4, 

                              random_state=1)

tree.fit(X_train, y_train)



X_combined = np.vstack((X_train, X_test))

y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, 

                      classifier=tree, test_idx=range(105, 150))



plt.xlabel('petal length [cm]')

plt.ylabel('petal width [cm]')

plt.legend(loc='upper left')

plt.tight_layout()

#plt.savefig('images/03_20.png', dpi=300)

plt.show()
!pip install pydotplus
from pydotplus import graph_from_dot_data

from sklearn.tree import export_graphviz



dot_data = export_graphviz(tree,

                           filled=True, 

                           rounded=True,

                           class_names=['Setosa', 

                                        'Versicolor',

                                        'Virginica'],

                           feature_names=['petal length', 

                                          'petal width'],

                           out_file=None) 

graph = graph_from_dot_data(dot_data) 

#graph.write_png('tree.png') 
Image(graph.create_png())