# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from skimage.feature import hog

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
X_full = train.iloc[:,1:]

y_full = train.iloc[:,:1]

x_train, x_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.15)

xt_non, xte_non, yt_non, yte_non = train_test_split(X_full, y_full, test_size = 0.15)
#reshape and normalise the dataset

x_train = x_train.values.reshape(-1, 28, 28).astype('float32') / 255.

x_test = x_test.values.reshape(-1, 28, 28).astype('float32') / 255.

test_set = test.values.reshape(-1, 28, 28).astype('float32') / 255.
def get_features(data):

    features = []

    for image in data:

        feat, hog_image = hog(image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(1, 1), block_norm='L2-Hys',\

                              visualize=True, multichannel=False)

        features.append(feat)

    return features
tr_feat = get_features(x_train)

ts_feat = get_features(x_test)

test_set = get_features(test)
knn_classifier = KNeighborsClassifier(n_neighbors=10, weights='uniform', n_jobs=4)

knn_classifier.fit(tr_feat, y_train['label'].values)

#y_pred = knn_classifier.predict(ts_feat)

#accuracy_score(y_test, y_pred)
log_clf = LogisticRegression()

log_clf.fit(xt_non, yt_non)
svc_clf = svm.SVC()

svc_clf.fit(xt_non, yt_non)

#svc_clf.score(xte_non, yte_non)
results = knn_classifier.predict(test_set)



results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_digit_knn.csv",index=False)
results = svc_clf.predict(test)



results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_digit_svm.csv",index=False)
results = log_clf.predict(test)



results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_digit_log.csv",index=False)