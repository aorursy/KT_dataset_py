# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score

from sklearn.metrics import accuracy_score, classification_report



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data =  pd.read_csv('/kaggle/input/malicious-and-benign-websites/dataset.csv')

data.head()
data.describe(include='all')
data = data.drop(labels=['URL','WHOIS_REGDATE','WHOIS_UPDATED_DATE'], axis='columns')
print(data.isnull().sum())



data[pd.isnull(data).any(axis='columns')]
processed_data =  data.interpolate()

print(processed_data.isnull().sum())
max_value = processed_data['SERVER'].value_counts().idxmax()



print('Highest frequency value:',max_value)



processed_data['SERVER'].fillna(max_value, inplace=True)



print(processed_data.isnull().sum())
knn_data = pd.get_dummies(processed_data, prefix_sep="_")

knn_data.head()
X = knn_data.drop(labels='Type', axis='columns')

y = knn_data['Type']

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



print("Training size: %d" % len(y_train))

print("Test size    : %d" % len(y_test))

from sklearn.neighbors import KNeighborsClassifier

k=10



clf = KNeighborsClassifier(n_neighbors=k, p=2, weights='distance')

clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)

print("Print results for 30 test data points:")

print("Predicted labels: ", y_pred[30:60])

print("Ground truth    : ", y_test.to_numpy()[30:60])



print("Accuracy of %d NN: %.2f %%" % (k, 100 * accuracy_score(y_test.to_numpy(), y_pred)))

print('Classification Report:\n{}\n'.format(classification_report(y_test.to_numpy(),clf.predict(X_test))))

y_pred_proba = clf.predict_proba(X_test)

print(y_pred_proba[30:60]*100)
nb_data = pd.get_dummies(processed_data, prefix_sep="_")

nb_data.head()
X = nb_data.drop(labels='Type', axis='columns')

y = nb_data['Type']

X.head()
X_train, X_test, y_train, y_test = train_test_split(nb_data, y, test_size=0.3)



print("Training size: %d" % len(y_train))

print("Test size    : %d" % len(y_test))

from sklearn.naive_bayes import GaussianNB

nb_clf= GaussianNB()

nb_clf.fit(X_train, y_train)



y_pred = nb_clf.predict(X_test)



print("Print results for 30 test data points:")

print("Predicted labels: ", y_pred[30:60])

print("Ground truth    : ", y_test.to_numpy()[30:60])



print("Accuracy of GNB: %.2f %%" % ( 100 * accuracy_score(y_test.to_numpy(), y_pred)))

print('Classification Report:\n{}\n'.format(classification_report(y_test.to_numpy(),nb_clf.predict(X_test))))
from sklearn.naive_bayes import MultinomialNB

nb_clf= MultinomialNB()

nb_clf.fit(X_train, y_train)



y_pred = nb_clf.predict(X_test)



print("Print results for 30 test data points:")

print("Predicted labels: ", y_pred[30:60])

print("Ground truth    : ", y_test.to_numpy()[30:60])



print("Accuracy of MNB: %.2f %%" % ( 100 * accuracy_score(y_test.to_numpy(), y_pred)))

print('Classification Report:\n{}\n'.format(classification_report(y_test.to_numpy(),nb_clf.predict(X_test))))
from sklearn.naive_bayes import BernoulliNB

nb_clf= BernoulliNB()

nb_clf.fit(X_train, y_train)



y_pred = nb_clf.predict(X_test)



print("Print results for 30 test data points:")

print("Predicted labels: ", y_pred[30:60])

print("Ground truth    : ", y_test.to_numpy()[30:60])



print("Accuracy of BNB: %.2f %%" % ( 100 * accuracy_score(y_test.to_numpy(), y_pred)))

print('Classification Report:\n{}\n'.format(classification_report(y_test.to_numpy(),nb_clf.predict(X_test))))
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaled_data  = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.4)



print("Training size: %d" % len(y_train))

print("Test size    : %d" % len(y_test))
from sklearn.naive_bayes import MultinomialNB

nb_clf= MultinomialNB()

nb_clf.fit(X_train, y_train)



y_pred = nb_clf.predict(X_test)



print("Print results for 30 test data points:")

print("Predicted labels: ", y_pred[30:60])

print("Ground truth    : ", y_test.to_numpy()[30:60])



print("Accuracy of MNB: %.2f %%" % ( 100 * accuracy_score(y_test.to_numpy(), y_pred)))

print('Classification Report:\n{}\n'.format(classification_report(y_test.to_numpy(),nb_clf.predict(X_test))))
import matplotlib.pyplot as plt

import time

from sklearn.model_selection import KFold



models = []

models.append(('KNN', KNeighborsClassifier(n_neighbors=k, p=2, weights='distance'), 0))

models.append(('GNB', GaussianNB(), 0))

models.append(('MNB', MultinomialNB(), 0))

models.append(('BNB', BernoulliNB(), 0))

# models with normalization on numerical columns

models.append(('KNN-S', KNeighborsClassifier(n_neighbors=k, p=2, weights='distance'), 1))

models.append(('GNB-S', GaussianNB(), 1))

models.append(('MNB-S', MultinomialNB(), 1))

models.append(('BNB-S', BernoulliNB(), 1))



results = []

names = []

run_times = []

scoring = 'accuracy'

for name, model, scaler in models:

    start = time.time()

    kfold = KFold(n_splits=10, random_state=7)

    if(scaler==1):

        scaler = MinMaxScaler()

        scaled_X  = scaler.fit_transform(X)

        cv_results = cross_val_score(model, scaled_X, y, cv=10, scoring=scoring)

    else:

        cv_results = cross_val_score(model, X, y, cv=10, scoring=scoring)

    stop= time.time()

    run_times.append(stop-start)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

    



print( "Run times: %s" % (run_times))

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Accuracy Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()





y_pos = np.arange(len(names))

plt.bar(y_pos, run_times, align='center', alpha=0.5)

plt.xticks(y_pos, names)

plt.title('Time Comparison')

plt.show()