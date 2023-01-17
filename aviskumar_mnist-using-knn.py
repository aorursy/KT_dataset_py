import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import neighbors

from sklearn.metrics import  confusion_matrix

from sklearn.utils.multiclass import unique_labels
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
X_train=np.reshape(x_train,(60000,784))

X_test=np.reshape(x_test,(10000,784))
print(len(X_train),len(X_train[0]))

print(len(X_test),len(X_test[0]))
print(len(y_test))

print(len(y_train))
idx=[]

for i in range(60000):

  if (y_train[i] == 2) | (y_train[i] == 6) | (y_train[i] == 8) :

    idx.append(i)



print(len(idx))



X=X_train[idx]

y=y_train[idx]
X.shape
y.shape
knn=neighbors.KNeighborsClassifier(n_neighbors=7)
knn.fit(X,y)
test_idx=[]

for i in range(10000):

  if (y_test[i] == 2) | ((y_train[i] == 6) | ((y_train[i] == 8))):

    test_idx.append(i)



print(len(test_idx))
X_test1=X_test[test_idx]

y_test1=y_test[test_idx]
X_test1.shape
y_pred=knn.predict(X_test1)
y_pred
knn.score(X_test1,y_test1)
import itertools
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion Matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    

    

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks=np.arange(len(classes))

    plt.xticks(tick_marks,classes,rotation=45)

    plt.yticks(tick_marks,classes)



   



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[0])) :

            plt.text(j, i, format(cm[i, j], fmt),

                    horizontalalignment="center", 

                    color="white" if cm[i, j] > thresh else "black")

    

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cm=metrics.confusion_matrix(y_test1,y_pred)
print(y_pred,y_test1)
plot_confusion_matrix(cm,["2","6","8"])