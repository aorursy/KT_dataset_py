import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn import svm

from sklearn.decomposition import PCA



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/Kannada-MNIST/train.csv")

test  = pd.read_csv("../input/Kannada-MNIST/test.csv")

submission  = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

val= pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
test.rename(columns={'id':'label'}, inplace=True)

test.head()
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:, 1:], train.iloc[:, 0], test_size=0.2)
# scaler = RobustScaler()

# X_train = scaler.fit_transform(X_train)

# X_test = scaler.transform(X_test)
pca = PCA(n_components=0.7,whiten=True)

X_train_PCA = pca.fit_transform(X_train)

X_test_PCA = pca.transform(X_test)
sv = svm.SVC(kernel='rbf', C=9, gamma='auto')

sv.fit(X_train_PCA , y_train)
y_pred = sv.predict(X_test_PCA)
from sklearn.metrics import accuracy_score

accuracy_score(y_pred,y_test)
test_x = test.values[:,1:]

test_x = pca.transform(test_x)
pred = sv.predict(test_x)
submission['label'] = pred

submission.to_csv('submission.csv', index=False)