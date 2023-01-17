import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import datetime 

import matplotlib.pyplot as plt

%matplotlib inline

import warnings  

warnings.filterwarnings('ignore')
pandas_train = pd.read_csv("../input/train.csv")

pandas_train = pandas_train.drop(["Id"], axis=1)

columns = pandas_train.axes[1]

data_train = np.array(pandas_train)

X = data_train[:,:-1]

Y = data_train[:,-1]
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)
start_time = datetime.datetime.now()

clf = SVC(C=1, kernel='rbf')

clf.fit(X_train, Y_train)

print("SVC time execution(kernel='rbf' and C=1 ):",

     datetime.datetime.now() - start_time)



prediction = clf.predict(X_train)

acc_score_train = accuracy_score(Y_train, prediction)

prediction = clf.predict(X_valid)

acc_score_valid = accuracy_score(Y_valid, prediction)

print("Train accuracy = {0}, Test accuracy = {1}".format(acc_score_train, acc_score_valid))
X_train_norm = preprocessing.normalize(X_train, norm='l2')

X_valid_norm = preprocessing.normalize(X_valid, norm='l2')
print("\nSVC with data normalization:")

start_time = datetime.datetime.now()

clf = SVC(C=1, kernel='rbf')

clf.fit(X_train_norm, Y_train)

print("SVC time execution(kernel='rbf' and C=1 ):",

     datetime.datetime.now() - start_time)



prediction = clf.predict(X_train_norm)

acc_score_train = accuracy_score(Y_train, prediction)

prediction = clf.predict(X_valid_norm)

acc_score_valid = accuracy_score(Y_valid, prediction)

print("Train accuracy = {0}, Test accuracy = {1}".format(acc_score_train, acc_score_valid))
X_train_scale = preprocessing.scale(X_train)

X_valid_scale = preprocessing.scale(X_valid)
print("\nSVC with scaling:")

start_time = datetime.datetime.now()

clf = SVC(C=1, kernel='rbf')

clf.fit(X_train_scale, Y_train)

print("SVC time execution(kernel='rbf' and C=1 ):",

     datetime.datetime.now() - start_time)



prediction = clf.predict(X_train_scale)

acc_score_train = accuracy_score(Y_train, prediction)

prediction = clf.predict(X_valid_scale)

acc_score_valid = accuracy_score(Y_valid, prediction)

print("Train accuracy = {0}, Test accuracy = {1}".format(acc_score_train, acc_score_valid))

acc_train, acc_valid = [], []

power = [i for i in range(-6, 6)]

for i in power:

    start_time = datetime.datetime.now()

    clf = SVC(C=10**i, kernel='rbf')

    clf.fit(X_train, Y_train)

    print("SVC time execution(kernel='rbf' and C={0}):".format(10**i),

          datetime.datetime.now() - start_time)

    

    prediction = clf.predict(X_train)

    acc_score_train = accuracy_score(Y_train, prediction)

    acc_train.append(acc_score_train)

    prediction = clf.predict(X_valid)

    acc_score_valid = accuracy_score(Y_valid, prediction)

    acc_valid.append(acc_score_valid)

    print("Train accuracy = {0}, Test accuracy = {1}".format(acc_score_train, acc_score_valid))
plt.plot(power, acc_train)

plt.plot(power, acc_valid)

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('C^x')

plt.legend(['train', 'valid'], loc='upper left')

plt.show()