# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Load libraries
from sklearn.model_selection import train_test_split
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
X_train = df_train.ix[:, 1:].values / 255.0
y_train = df_train.ix[:, 0].values
X_test = df_test.values / 255.0
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
#from tensorflow.contrib import skflow
import tensorflow as tf
#from tensorflow.contrib import learn
from tensorflow.contrib import skflow

from sklearn import metrics
clf = skflow.TensorFlowDNNClassifier(
    hidden_units=[500, 500], steps=5000, learning_rate=0.1, 
    batch_size=10, n_classes=10)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)
metrics.accuracy_score(y_train, y_train_pred)
y_test_pred = clf.predict(X_test)
submission = pd.DataFrame({
        "ImageId": np.arange(1, df_test.shape[0]+1),
        "Label": y_test_pred
    })
submission.to_csv("sub.csv", index=False)