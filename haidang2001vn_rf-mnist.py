import pandas as pd

import numpy as np



import gzip, pickle
import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.linear_model import SGDClassifier 



from sklearn.ensemble import BaggingClassifier 

from sklearn.tree import DecisionTreeClassifier



from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



from sklearn.metrics import roc_curve

from sklearn.metrics import auc
def plotConfusionMatrix(y_true, y_pred, classes,

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
!ls ../input/mnistpklgz

with gzip.open("../input/mnistpklgz/mnist.pkl.gz","rb") as ff :

    u = pickle._Unpickler( ff )

    u.encoding = "latin1"

    train, val, test = u.load()
print( train[0].shape, train[1].shape )
print( val[0].shape, val[1].shape )
print( test[0].shape, test[1].shape )
some_digit = train[0][0]

some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="lanczos")

plt.axis("off")

plt.show()
train[1][0]
X_train = train[0]

X_val = val[0]

X_test = test[0]
y_train = train[1].astype(np.uint8)

y_val = val[1].astype(np.uint8)

y_test = test[1].astype(np.uint8)
from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier

import pandas as pd



clf = DecisionTreeClassifier(random_state=0)

iris = load_iris()

iris_pd = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

clf = clf.fit(iris_pd, iris.target)
print(dict(zip(iris_pd.columns, clf.feature_importances_)))
bagClf = BaggingClassifier(

    DecisionTreeClassifier(), n_estimators=100,

    max_samples=100, bootstrap=True

)

bagClf.fit(X_train, y_train)

y_pred = bagClf.predict(X_test)
set(y_test)
unique_labels(y_test, y_pred)
classesName = np.array(range(10))
featureImportances = np.mean([

    tree.feature_importances_ for tree in bagClf.estimators_

], axis=0)
baggingPixelImportances = featureImportances.reshape(28, 28)

fig, ax = plt.subplots()

im = ax.imshow(baggingPixelImportances, interpolation="lanczos", cmap=mpl.cm.afmhot)

ax.figure.colorbar(im, ax=ax)

plt.show()
## Confusion matrix

plotConfusionMatrix(y_test, y_pred, classesName)
rndClf = RandomForestClassifier(

    n_estimators=100, max_leaf_nodes=16, n_jobs=-1

)

rndClf.fit(X_train, y_train)

y_pred_rf = rndClf.predict(X_test)
rndClf.feature_importances_
pixelImportance = rndClf.feature_importances_.reshape(28, 28)
fig, ax = plt.subplots()

im = ax.imshow(pixelImportance, interpolation="lanczos", cmap=mpl.cm.afmhot)

ax.figure.colorbar(im, ax=ax)

plt.show()
## Confusion matrix

plotConfusionMatrix(y_pred_rf, y_pred, classesName)