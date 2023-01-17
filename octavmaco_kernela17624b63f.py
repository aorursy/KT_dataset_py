# Importul bibliotecilor necesare pentru operații matematice, manipularea datelor și reprezentarea acestora

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from scipy.io import loadmat



%matplotlib inline
# Stocarea căilor către fișiere în variabile 

DATAPATH_1 = ('../input/ex6data1.mat')

DATAPATH_2 = ('../input/ex6data2.mat')

DATAPATH_3 = ('../input/ex6data3.mat')

DATA_SPAM_TRAIN = ('../input/spamTrain.mat')

DATA_SPAM_TEST = ('../input/spamTest.mat')
def plot_data(X, y, xlabel, ylabel, pos_label, neg_label, xmin, xmax, ymin, ymax, axes=None):

    plt.rcParams['figure.figsize'] = (20., 14.)

    

    pos = y[:, 0] == 1

    neg = y[:, 0] == 0

    

    if axes == None:

        axes = plt.gca()

    

    axes.scatter(X[pos][:,0], X[pos][:,1], marker='o', c='#003f5c', s=50, linewidth=2, label=pos_label)

    axes.scatter(X[neg][:,0], X[neg][:,1], marker='o', c='#ffa600', s=50, linewidth=2, label=neg_label)

    

    axes.set_xlim([xmin, xmax])

    axes.set_ylim([ymin, ymax])

    

    axes.set_xlabel(xlabel, fontsize=12)

    axes.set_ylabel(ylabel, fontsize=12)

    

    axes.legend(bbox_to_anchor=(1,1), fancybox=True)
data1 = loadmat(DATAPATH_1)



X = data1['X']

y = data1['y']
plot_data(X, y, 'X', 'y', 'pozitiv', 'negativ', 0, 4.2, 0, 5)
# Importul obiectului sm (Support Vector)

from sklearn import svm



# Declararea și antrenarea algoritmului pentru mașini cu suport vectorial având parametrul de optimizare C=1

clf = svm.SVC(kernel='linear', C=1, decision_function_shape = 'ovr')

clf.fit(X, y.ravel())



# Reprezentarea grafică a datelor

plot_data(X, y, 'X', 'y', 'pozitiv', 'negativ', 0, 4.2, 0, 5)



# Reprezentarea grafică a hiperplanului

x_1, x_2 = np.meshgrid(np.arange(0.0, 5.0, 0.01), np.arange(0.0, 5.0, 0.01))

Z = clf.predict(np.c_[x_1.ravel(), x_2.ravel()])

Z = Z.reshape(x_1.shape)

plt.contour(x_1, x_2, Z, [0.5], colors='b')
#Use C=100

clf100 = svm.SVC(kernel='linear', C=100.0, decision_function_shape='ovr')

clf100.fit(X, y.ravel())



#Plot data

plot_data(X, y, 'X', 'y', 'pozitiv', 'negativ', 0, 4.2, 0, 5)



#Plot hyperplane



x_1, x_2 = np.meshgrid(np.arange(0.0, 5.0, 0.01), np.arange(0.0, 5.0, 0.01))

Z = clf100.predict(np.c_[x_1.ravel(), x_2.ravel()])

Z = Z.reshape(x_1.shape)

plt.contour(x_1, x_2, Z, [0.5], colors='b')
data2 = loadmat(DATAPATH_2)



X_2 = data2['X']

y_2 = data2['y']



plot_data(X_2, y_2, 'X', 'y', 'pozitiv', 'negativ', 0, 1.0, 0.38, 1)
sigma = 0.1

gamma = 1/(2 * sigma**2)



clfg = svm.SVC(kernel='rbf', gamma=gamma, C=1.0, decision_function_shape='ovr')

clfg.fit(X_2, y_2.ravel())



plot_data(X_2, y_2, 'X', 'y', 'pozitiv', 'negativ', 0, 1, 0.38, 1)



x_1, x_2 = np.meshgrid(np.arange(0.0, 1.0, 0.003), np.arange(0.38, 1.0, 0.003))

Z = clfg.predict(np.c_[x_1.ravel(), x_2.ravel()])

Z = Z.reshape(x_1.shape)

plt.contour(x_1, x_2, Z, [0.5], colors='b')
data3 = loadmat(DATAPATH_3)



X_3 = data3['X']

y_3 = data3['y']



plot_data(X_3, y_3, 'X', 'y', 'pozitiv', 'negativ', -0.55, 0.35, -0.8, 0.6)
sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]



errors = list()

sigma_c = list()



for each in sigma:

    for each_c in C:

        clf = svm.SVC(kernel='rbf', gamma = 1/(2*(each**2)), C=each_c, decision_function_shape='ovr')

        clf.fit(X_3, y_3.ravel())

        errors.append(clf.score(data3['Xval'], data3['yval'].ravel()))

        sigma_c.append((each, each_c))
index = np.argmax(errors)



sigma_max, c_max = sigma_c[index]



print('Valoarea optimă a lui sigma este: {}'.format(sigma_max))

print('Valoarea optimă a lui C este: {}'.format(c_max))
sigma= 0.1

gamma = 1/(2*(sigma**2))



optimal_clf = svm.SVC(kernel='rbf', gamma=gamma, C=1.0, decision_function_shape='ovr')

optimal_clf.fit(X_3, y_3.ravel())



plot_data(X_3, y_3, 'X', 'y', 'pozitiv', 'negativ', -0.55, 0.35, -0.8, 0.6)



x_1, x_2 = np.meshgrid(np.arange(-0.6, 0.4, 0.004), np.arange(-0.8, 0.6, 0.004))

Z = optimal_clf.predict(np.c_[x_1.ravel(), x_2.ravel()])

Z = Z.reshape(x_1.shape)

plt.contour(x_1, x_2, Z, [0.5], colors='b')
spam_train = loadmat(DATA_SPAM_TRAIN)

spam_test = loadmat(DATA_SPAM_TEST)



C = 0.1



X_train = spam_train['X']

y_train = spam_train['y']



X_test = spam_test['Xtest']

y_test = spam_test['ytest']



clf_spam = svm.SVC(kernel = 'linear', C = 0.1, decision_function_shape = 'ovr')

clf_spam.fit(X_train, y_train.ravel())



train_acc = clf_spam.score(spam_train['X'], spam_train['y'].ravel())

test_acc = clf_spam.score(X_test, y_test.ravel())



print(f'Acuratețea pe setul de antrenare = {train_acc * 100}')

print(f'Acuratețea pe setul de test = {test_acc * 100}')