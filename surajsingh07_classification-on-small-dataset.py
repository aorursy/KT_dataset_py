import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.datasets import make_blobs

X1, y1 = make_blobs(n_samples=500, n_features=2, centers=2, cluster_std=1.5, random_state=43)



plt.scatter(X1[:,0], X1[:,1], c=y1)

plt.xlabel('feature_1')

plt.ylabel('feature_2')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=43)
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier().fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('training accuracy: {}'.format(knn.score(X_train,y_train)))

print('test accuracy: {}'.format(knn.score(X_test, y_test)))

print('confusion matrix on test set: \n{}'.format(confusion_matrix(y_test, y_pred)))



fig, axs = plt.subplots(ncols=2, figsize=(20,5))

axs[0].scatter(X_test[:,0], X_test[:,1], c=y_test)

axs[0].set_title('Actual')

axs[0].set_xlabel('feature_1')

axs[0].set_ylabel('feature_2')



axs[1].scatter(X_test[:,0], X_test[:,1], c=y_pred)

axs[1].set_title('Predicted')

axs[1].set_xlabel('feature_1')

axs[1].set_ylabel('feature_2')



plt.show()
train_accuracy = []

test_accuracy = []

for i in range(1,51):

    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)

    train_accuracy.append(knn.score(X_train,y_train))

    test_accuracy.append(knn.score(X_test, y_test))

    

plt.figure(figsize=(20,5))

sns.set_style('whitegrid')

plt.plot(range(1,51),train_accuracy, label='train')

plt.plot(range(1,51),test_accuracy, label='test')

plt.xlabel('n_neighbors')

plt.ylabel('accuracy')

plt.xticks(np.arange(min(range(1,51)), max(range(1,51))+1, 1.0))

plt.legend()

plt.show()
from sklearn.linear_model import LogisticRegression



lgr = LogisticRegression().fit(X_train, y_train)

y_pred = lgr.predict(X_test)

print('training accuracy: {}'.format(lgr.score(X_train,y_train)))

print('test accuracy: {}'.format(lgr.score(X_test, y_test)))

print('confusion matrix on test set: \n{}'.format(confusion_matrix(y_test, y_pred)))



fig, axs = plt.subplots(ncols=2, figsize=(20,5))

axs[0].scatter(X_test[:,0], X_test[:,1], c=y_test)

axs[0].set_title('Actual')

axs[0].set_xlabel('feature_1')

axs[0].set_ylabel('feature_2')



axs[1].scatter(X_test[:,0], X_test[:,1], c=y_pred)

axs[1].set_title('Predicted')

axs[1].set_xlabel('feature_1')

axs[1].set_ylabel('feature_2')



plt.show()
train_accuracy = []

test_accuracy = []

C_values = [0.001, 0.01, 0.1, 1, 100]

for i in C_values:

    lgr = LogisticRegression(C=i).fit(X_train, y_train)

    train_accuracy.append(lgr.score(X_train, y_train))

    test_accuracy.append(lgr.score(X_test, y_test))



score_table = pd.DataFrame({'C':C_values,'train_accuacy':train_accuracy,'test_accuracy':test_accuracy}).set_index('C')

score_table.transpose()
lgr001 = LogisticRegression(C=0.001).fit(X_train, y_train)

lgr01 = LogisticRegression(C=0.01).fit(X_train, y_train)

lgr1 = LogisticRegression(C=0.1).fit(X_train, y_train)

lgr_1 = LogisticRegression(C=1).fit(X_train, y_train)

lgr_100 = LogisticRegression(C=100).fit(X_train, y_train)



plt.figure(figsize=(15,5))

plt.plot(lgr001.coef_.T, 'o', label="C=0.001")

plt.plot(lgr01.coef_.T, 'p', label="C=0.01")

plt.plot(lgr1.coef_.T, 'D', label="C=0.1")

plt.plot(lgr_1.coef_.T, '^', label="C=1")

plt.plot(lgr_100.coef_.T, 'v', label="C=100")

plt.xticks(range(X_train.shape[1]), ['feature_1','feature_2'])

plt.hlines(0, 0, 2)

plt.xlabel("Coefficient index")

plt.ylabel("Coefficient magnitude")

plt.legend()

plt.show()