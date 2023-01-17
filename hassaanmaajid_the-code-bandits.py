# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import os

import numpy as np
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.info()
test.info()
from sklearn.model_selection import train_test_split
X_train_split = train.drop(['label'], axis=1).copy()

y_train_split = train['label'].copy()



X_train, X_validation, y_train, y_validation = train_test_split(X_train_split, y_train_split, test_size=0.1, random_state=42)



del X_train_split, y_train_split



print("Training Features:", X_train.shape)

print("Training Labels:", y_train.shape)

print("Validation Features:", X_validation.shape)

print("Validation Labels:", y_validation.shape)

print("Test Features:", test.shape)
sample_digit = X_train.iloc[2000] # a random instance

sample_digit_image = sample_digit.values.reshape(28, 28) # reshape it from (784,) to (28,28)

plt.imshow(sample_digit_image, # plot it as an image

           cmap = matplotlib.cm.binary,

           interpolation="nearest")

plt.axis("off")

plt.show()
X_train = X_train / 255.0

X_validation = X_validation / 255.0

X_train.head()
from sklearn.neural_network import MLPClassifier
mlp_clf = MLPClassifier(random_state=42)

mlp_clf.fit(X_train, y_train)
from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier()

kn_clf.fit(X_train, y_train)
from skimage.feature import hog

from sklearn.svm import LinearSVC
list_hog_fd = []

for feature in X_train.values:

    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))

    list_hog_fd.append(fd)

hog_features = np.array(list_hog_fd, 'float64')
clf = LinearSVC()

clf.fit(hog_features, y_train)
from sklearn.metrics import accuracy_score
knn_prediction = kn_clf.predict(X_validation)

print("KNN Accuracy:", accuracy_score(y_true=y_validation ,y_pred=knn_prediction))
mlp_prediction = mlp_clf.predict(X_validation)

print("MLP Accuracy:", accuracy_score(y_true=y_validation ,y_pred=mlp_prediction))
list_hog_fd = []

for feature in X_validation.values:

    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))

    list_hog_fd.append(fd)

hog_features = np.array(list_hog_fd, 'float64')





svc_bw_prediction = clf.predict(hog_features)

print("SVC HOG Accuracy:", accuracy_score(y_true=y_validation ,y_pred=svc_bw_prediction))