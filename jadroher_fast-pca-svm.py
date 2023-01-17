import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.pyplot as plt

import itertools

%matplotlib inline
data = pd.read_csv('../input/data.csv')
data.head()
list(data.columns)
Y = data['diagnosis']

X = data.drop(['id','diagnosis','Unnamed: 32'],axis=1)
X2 = X.values

Y2 = Y.apply(lambda x: 0 if x=='B' else 1).values
print(X2)

print(X2.shape)
print(Y2)

print(Y2.shape)
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X2)
X_std
from sklearn.decomposition import PCA
pca = PCA()

pca.fit(X_std)
plt.plot(pca.explained_variance_ratio_)
pca2 = PCA(n_components = 6)

pca2.fit(X_std)

X_PCA = pca2.transform(X_std)
X_PCA
X_PCA.shape
from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn.metrics import recall_score
X_train, X_test, Y_train, Y_test = train_test_split(X_PCA, Y2, test_size=0.3)
X_train.shape
X_test.shape
clf = svm.SVC()

clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
# Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cnf_matrix = confusion_matrix(Y_test, Y_pred, labels =[0,1])

np.set_printoptions(precision=2)



plot_confusion_matrix(cnf_matrix, classes=['B','M'])
recall_score(Y_test, Y_pred, labels =[0,1])