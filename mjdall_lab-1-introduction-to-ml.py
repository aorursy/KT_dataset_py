import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
def print_predicted_stats(predicted, y_test):

    hit = 0

    miss = 0

    for actual, prediction in zip(y_test, predicted):

        if actual == prediction:

            hit += 1

            continue

        miss += 1

    percent = 100 - ((miss/hit) * 100)

    print('missed: %d\nhit: %d\n%d%s' % (miss, hit, percent, '% accuracy'))
input_csv_loc = '../input/wisconsin_breast_cancer.csv'

bccf = pd.read_csv(input_csv_loc)

# get rid of NaN entries, save prev and post counts for VERBOSE mode

bccf_clean = bccf.dropna()

bccf_clean.head()
import seaborn as sns

sns.pairplot(data=bccf_clean, hue='single', palette='Set2')
from sklearn.model_selection import train_test_split



# select all entries, then get from col1 and select up until the end, exlcuding it

x_dat = bccf_clean.iloc[:, 1:-1]

# select all entries, then get col 10

y_dat = bccf_clean.iloc[:, 10]



x_train, x_test, y_train, y_test = train_test_split(x_dat, y_dat, test_size=0.2, random_state=547839)



print('x_train: %s\ny_train: %s\nx_test: %s\ny_test: %s' % (

    x_train.shape, y_train.shape,

    x_test.shape, y_test.shape))
from sklearn.svm import SVC

model = SVC()

model.fit(x_train, y_train)
predicted = model.predict(x_test)

print_predicted_stats(predicted, y_test)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))
x_test.head()
"""

cols:

    thickness: 0,

    size: 1,

    shape: 2,

    adhesion: 3,

    single: 4,

    nuclei: 5,

    chromatin: 6,

    nucleoli: 7,

    mitosis: 8

my choice: [thickness, size, single, chromatin]

"""

# selecting rows to increase accuracy - eyeballed 99% accuracy

rows = [0, 1, 4, 6]

# experimenting getting bad accuracy, came out with 96%

# rows = [5, 4, 8, 1]

xsub_train = x_train.iloc[:, rows]

xsub_test = x_test.iloc[:, rows]

xsub_train.head()
second_model = SVC()

second_model.fit(xsub_train, y_train)
sub_predicted = second_model.predict(xsub_test)

print_predicted_stats(sub_predicted, y_test)
print(confusion_matrix(y_test, sub_predicted))
print(classification_report(y_test, sub_predicted))
import sklearn.neighbors

#when k value is 1

knn_1_model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=1)

knn_1_model.fit(x_train, y_train)
knn_1_predicted = knn_1_model.predict(x_test)

print_predicted_stats(knn_1_predicted, y_test)
print(confusion_matrix(y_test, knn_1_predicted))
print(classification_report(y_test, knn_1_predicted))
#when k value is 1

knn_5_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)

knn_5_model.fit(x_train, y_train)
knn_5_predicted = knn_5_model.predict(x_test)

print_predicted_stats(knn_5_predicted, y_test)
y_test.head()
print(confusion_matrix(y_test, knn_5_predicted))
print(classification_report(y_test, knn_5_predicted))