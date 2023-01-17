# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/digit-recognizer'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC  # Support Vector Classification
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

X_train_split = train.drop(['label'], axis=1).copy()
Y_train_split = train['label'].copy()

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_split, Y_train_split, test_size=0.1,
                                                                random_state=42)
del X_train_split, Y_train_split
sample_digit = X_train.iloc[2000]  # a random instance
sample_digit_image = sample_digit.values.reshape(28, 28)  # reshape it from (784,) to (28,28)
plt.imshow(sample_digit_image,  # plot it as an image
           cmap=matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()

X_train /= 255.0
X_validation /= 255.0
test /= 255.0
# Below lines calculate the best value of K to be used in KNN Classifier
value = len(X_validation)
value = int(pow(value, 0.25))
if value % 2 == 0:
    value += 1
print(value)

# Below lines predict the accuracy of the Digits using KNN Classifier
kn_clf = KNeighborsClassifier(n_neighbors=value)
kn_clf.fit(X_train, Y_train)

knn_prediction = kn_clf.predict(X_validation)
print("KNN Accuracy:", accuracy_score(y_true=Y_validation, y_pred=knn_prediction))
# Below lines predict the accuracy of the Digits using MLP Classifier
mlp_clf = MLPClassifier(hidden_layer_sizes=(300,), learning_rate_init=0.001, solver='adam',
                        random_state=42, verbose=True)
mlp_clf.fit(X_train, Y_train)

mlp_prediction = mlp_clf.predict(X_validation)
print("MLP Accuracy:", accuracy_score(y_true=Y_validation, y_pred=mlp_prediction))

# Below lines predict the accuracy of the Digits using SVM Classifier
svc_clf = SVC(gamma='auto', random_state=42, verbose=True)
svc_clf.fit(X_train, Y_train)

svc_prediction = svc_clf.predict(X_validation)
print("SVC Accuracy:", accuracy_score(y_true=Y_validation, y_pred=svc_prediction))
# Below lines predict the accuracy of the Digits using SVM-HOG Classifier
list_hog_fd = []
for feature in X_train.values:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

clf = LinearSVC()
clf.fit(hog_features, y_train)

list_hog_fd1 = []
for feature in X_validation.values:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    list_hog_fd1.append(fd)
hog_features = np.array(list_hog_fd1, 'float64')

svc_hog = clf.predict(hog_features)
print("SVC with HOG Accuracy:", accuracy_score(y_true=y_validation, y_pred=svc_hog))
# Below code creates a csv file for Kaggle submission on Test Data for KNN Classifier

final_prediction = kn_clf.predict(test)
submission = pd.DataFrame({"ImageId": list(range(1, len(final_prediction) + 1)),
                         "Label": final_prediction})
submission.to_csv("/kaggle/working/knn_mnist_submission.csv", index=False)
# Below code creates a csv file for Kaggle submission on Test Data for MLP Classifier

final_prediction = mlp_clf.predict(test)
submission = pd.DataFrame({"ImageId": list(range(1, len(final_prediction) + 1)),
                          "Label": final_prediction})
submission.to_csv("/kaggle/working/cnn_mnist_submission.csv", index=False)
# Below code creates a csv file for Kaggle submission on Test Data for SVM Classifier

final_prediction = svc_clf.predict(test)
submission = pd.DataFrame({"ImageId": list(range(1, len(final_prediction) + 1)),
                          "Label": final_prediction})
submission.to_csv("/kaggle/working/svm_mnist_submission.csv", index=False)
# Below code creates a csv file for Kaggle submission on Test Data for SVM-HOG Classifier

list_hog_fd2 = []
for feature2 in test.values:
   fd1 = hog(feature2.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
   list_hog_fd2.append(fd1)
hog_features2 = np.array(list_hog_fd2, 'float64')

final_prediction = clf.predict(hog_features2)
submission = pd.DataFrame({"ImageId": list(range(1, len(final_prediction) + 1)),
                         "Label": final_prediction})
submission.to_csv("/kaggle/working/svmHog_mnist_submission.csv", index=False)