import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
print('Shape of train data: {0}. Shape of test data: {1}'.format(train_data.shape, test_data.shape))
train_data.head()
test_data.head()
train_label = train_data['label']
train_label.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split



y = train_label.values

X = train_data.drop('label', 1).values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# It will take about 180 seconds in my Mac Book Pro (2.6 GHz Intel Core i5)

import time



start = time.perf_counter()



n_train = 100 #X.shape[0]

n_test = 10 #X_test.shape[0]



clf = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)

clf.fit(X[:n_train], y[:n_train])

score = clf.score(X_test[:n_test], y_test[:n_test])



end = time.perf_counter()

print('time: {0:.2f}; score: {1:.6f}'.format(end-start, score))
# It will take about 546 seconds in my Mac Book Pro (2.6 GHz Intel Core i5)

X_result = test_data.values



n_result = 100 #X.shape[0]

start = time.perf_counter()



y_result = clf.predict(X_result[:n_result])

print('time: {0:.2f}; y.shape={1}'.format(time.perf_counter() - start, y_result.shape))
# save results

# pd.DataFrame(y_result).to_csv('../result.csv', index=True, index_label=['ImageId'], header=['Label'])
# dump and load model

# from sklearn.externals import joblib

# joblib.dump(clf, 'knn.pkl');

# clf = joblib.load('digits_svm.pkl')