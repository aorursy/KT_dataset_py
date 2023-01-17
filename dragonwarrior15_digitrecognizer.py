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
# read the files and get X, y
import csv
f = open('../input/train.csv', 'r')
csv_reader_obj = csv.reader(f)
train_data = []
for row in csv_reader_obj:
    train_data.append(row)
f.close()
X = [train_data[i][1:] for i in range(1,len(train_data))]
y = [train_data[i][0] for i in range(1,len(train_data))]
# to check length of the X and y arrays
print(len(X))
print(len(X[0]))
print(len(train_data[0]))
print(len(y))
# split X and y into training and tst data set for training and evaluating the model
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# train a random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 20, max_depth = 10)
clf.fit(X_train, y_train)

# predict on X_test
y_test_pred = clf.predict(X_test)
# get the accuracies and relevant metrics
from sklearn.metrics import metrics
print('Accuracy Score : ' + str(metrics.accuracy_score(y_test, y_test_pred)))
print('Confusion Matrix : ' + str(metrics.confusion_matrix(y_test, y_test_pred)))
# import the test dataset on which we have to output the results of our model
f = open('../input/test.csv', 'r')
csv_reader_obj = csv.reader(f)
train_data = []
for row in csv_reader_obj:
    train_data.append(row)
X_test_dataset = [train_data[i][:] for i in range(1,len(train_data))]
# predict the classification for test dataset
y_test_dataset = clf.predict(X_test_dataset)
# write outputs to file
print('ImageId,Label')
for index, item in enumerate(y_test_dataset):
    print(str(index) + ',' + str(item))