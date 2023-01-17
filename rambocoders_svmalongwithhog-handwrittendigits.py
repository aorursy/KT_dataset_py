# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import the modules
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from keras.datasets import mnist
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# Use MNIST handwriting dataset
mnist = tf.keras.datasets.mnist

# Prepare data for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
features = np.array(x_train, 'int16')
labels = np.array(y_train, 'int')

# Function to find the hog features list

def hog_feature (TrainData):
    list_hog_fd = []
    visualise = False
    for feature in TrainData:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
        list_hog_fd.append(fd)
    hog_features = np.array(list_hog_fd, 'float64')
    return hog_features


hog_features = hog_feature(features)
testFeature = hog_feature(x_test)



clf = SVC( kernel="rbf")


clf.fit(hog_features, labels)
y_pred = clf.predict(testFeature)
df_submission = pd.DataFrame([range(1, x_test.size),y_pred],["ImageId","Label"]).transpose()
df_submission.to_csv("submission.csv",index=False)

# metrics
print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")
print(metrics.confusion_matrix(y_test, y_pred), "\n")

print (clf.score(hog_features, labels))

joblib.dump(clf, "digits_cls.pkl", compress=3)