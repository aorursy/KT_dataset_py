import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
digits_train = pd.read_csv('../input/train.csv')
digits_train.head()
digits_train.isnull().any().sum()
train_X = digits_train.drop(['label'], axis = 1).values
train_X.shape
train_y = digits_train.iloc[:,0]
train_y.shape
plt.imshow(train_X[15].reshape(28,28), interpolation='bilinear', cmap='gray');
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
knn = KNeighborsClassifier(n_neighbors=1)
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size = 0.2)
knn.fit(X_train, y_train)
predicted = knn.predict(X_test)
np.mean(predicted==y_test)
digits_test = pd.read_csv('../input/test.csv')
digits_test.head()
predicted = knn.predict(digits_test)
index = np.arange(1,28001)
submission = pd.Series(predicted, index = index, name = 'Label')
submission.to_csv("submission.csv", header=True , index_label='ImageId')