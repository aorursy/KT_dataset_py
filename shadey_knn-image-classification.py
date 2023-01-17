import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
X, y = train.drop(labels = ["label"],axis = 1).as_matrix(), train["label"]

X.shape
y.shape
%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

some_digit = X[71]

some_digit_show = plt.imshow(X[71].reshape(28,28), cmap=mpl.cm.binary)

y[71]
y = y.astype(np.uint8)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=4, n_jobs=-1)

knn_clf.fit(X_train, y_train)
y_knn_pred = knn_clf.predict([some_digit])

y_knn_pred
y_knn_pred = knn_clf.predict(X_test)



from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_knn_pred)
from scipy.ndimage.interpolation import shift
def shift_image(image, dx, dy):

    image = image.reshape((28, 28))

    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")

    return shifted_image.reshape([-1])
X_train_augmented = [image for image in X_train]

y_train_augmented = [label for label in y_train]



for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):

    for image, label in zip(X_train, y_train):

        X_train_augmented.append(shift_image(image, dx, dy))

        y_train_augmented.append(label)



X_train_augmented = np.array(X_train_augmented)

y_train_augmented = np.array(y_train_augmented)
shuffle_idx = np.random.permutation(len(X_train_augmented))

X_train_augmented = X_train_augmented[shuffle_idx]

y_train_augmented = y_train_augmented[shuffle_idx]
knn_clf.fit(X_train_augmented, y_train_augmented)
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)

confusion_matrix(y_train, y_train_pred)
knn_pred = knn_clf.predict(X_test)

accuracy_score(y_test, knn_pred)
knn_results=pd.DataFrame({"ImageId": list(range(1,len(knn_pred)+1)),

                         "Label": knn_pred})

knn_results.to_csv("KNN_clf.csv", index=False, header=True)