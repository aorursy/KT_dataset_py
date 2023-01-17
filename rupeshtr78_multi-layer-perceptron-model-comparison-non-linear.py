from sklearn import datasets

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



np.random.seed(0)



X, y = datasets.make_circles(n_samples=1000, factor=.3, noise=0.15)
plt.figure()

plt.subplots(1,1,figsize=(5,5))

# plt.subplot(1,2,3, projection='polar')

plt.title("Original Data")

reds = y == 0

blues = y == 1



plt.scatter(X[reds, 0], X[reds, 1], c="red",

            s=20, edgecolor='k')

plt.scatter(X[blues, 0], X[blues, 1], c="blue",

            s=20, edgecolor='k')

plt.xlabel("$x_1$")

plt.ylabel("$x_2$")
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=1, stratify=y)
k = np.bincount(y)

m = np.bincount(y_test)

n = np.bincount(y_train)

fig = plt.figure()

ax1 = fig.add_axes([0, 0, .5, .5], aspect=1)

ax1.pie(m,labels=m)

ax2 = fig.add_axes([.5, .0, .5, .5], aspect=1)

ax2.pie(n,labels=n)

ax3 = fig.add_axes([1, .0, .5, .5], aspect=1)

ax3.pie(k,labels=k)



ax1.set_title('y_test')

ax2.set_title('y_train')

ax3.set_title('y')

plt.show()

pass
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(X_train)

X_train_scale = sc.transform(X_train)

X_test_scale= sc.transform(X_test)
from sklearn.linear_model import Perceptron



ppn = Perceptron(eta0=0.1, random_state=1)

ppn.fit(X_train_scale, y_train)
y_pred = ppn.predict(X_test_scale)

print('Misclassified examples: %d' % (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score



print('Prediction Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('Training Accuracy: %.3f' % ppn.score(X_train_scale, y_train))
print('Test Accuracy: %.3f' % ppn.score(X_test_scale, y_test))
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X, y, clf=ppn, legend=2)
from sklearn.svm import SVC



svm = SVC(kernel='rbf',C=1 , gamma=100)

svm.fit(X_train_scale, y_train)
svc_y_pred = svm.predict(X_test_scale)

print('False Negative: %d' % (y_test != svc_y_pred).sum())
print('Prediction Accuracy SVC: %.3f' % accuracy_score(y_test, svc_y_pred))
print('Training Accuracy: %.3f' % svm.score(X_train_scale, y_train))
print('Test Accuracy: %.3f' % svm.score(X_test_scale, y_test))
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, svc_y_pred)

print(cm)
plot_decision_regions(X, y, clf=svm, legend=2)
fig = plt.figure(figsize=(10,8))

ax1 = fig.add_axes([0, 0, .5, .5], aspect=1)

plot_decision_regions(X_test_scale, y_test, clf=ppn, legend=2)

ax2 = fig.add_axes([.5, .0, .5, .5], aspect=1)

plot_decision_regions(X_test_scale, y_test, clf=svm, legend=2)





ax1.set_title('Perceptron')

ax2.set_title('SVC-RBF')



plt.show()

pass
from sklearn.neural_network import MLPClassifier



clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,

                   hidden_layer_sizes=(5, 2), random_state=1)



clf_mlp.fit(X_train_scale, y_train)
mlf_y_pred = clf_mlp.predict(X_test_scale)

print('False Negative: %d' % (y_test != mlf_y_pred).sum())
print('Prediction Accuracy SVC: %.3f' % accuracy_score(y_test, mlf_y_pred))
plot_decision_regions(X_test_scale, y_test, clf=clf_mlp, legend=2)
[coef.shape for coef in clf_mlp.coefs_]
clf_mlp.predict_proba(X_test_scale)
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB 

from sklearn.ensemble import RandomForestClassifier

import matplotlib.gridspec as gridspec

import itertools
clf_pct = Perceptron()

clf_rfc = RandomForestClassifier(random_state=1, n_estimators=100)

clf_nb = GaussianNB()

clf_svc = SVC(gamma=100)
gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(10,8))



labels = ['Perceptron', 'Random Forest', 'Naive Bayes', 'SVM']

for clf, lab, grd in zip([clf_pct, clf_rfc, clf_nb, clf_svc],

                         labels,

                         itertools.product([0, 1], repeat=2)):



    clf.fit(X, y)

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)

    plt.title(lab)

   



plt.show()