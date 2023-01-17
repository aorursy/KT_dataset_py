# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.describe()
test_df.describe()
train_data = train_df.iloc[:,0:2].values
train_labels = train_df.iloc[:,2].astype('int')

test_data = test_df.iloc[:,0:2].values
test_labels = test_df.iloc[:,2].astype('int')
plt.scatter(train_data[:,0], train_data[:,1], c=train_labels.apply(lambda x: 'red' if x == 0 else 'blue'))
plt.scatter(test_data[:,0], test_data[:,1], c=test_labels.apply(lambda x: 'red' if x == 0 else 'blue'))
#model = KNeighborsClassifier(n_neighbors=1)
#model = SVC()
model = GaussianNB()
model.fit(train_data, train_labels)
predictions = model.predict(test_data)
accuracy = accuracy_score(test_labels.values, predictions)

print("Accuracy Score is: {}".format(accuracy))

