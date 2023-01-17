!pip install -U cvxopt
from __future__ import print_function

import numpy as np 

import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from sklearn.svm import SVC



from cvxopt import matrix, solvers
np.random.seed(22)



means = [[2, 2], [4, 2]]

cov = [[.3, .2], [.2, .3]]

N = 20

X0 = np.random.multivariate_normal(means[0], cov, N) # class 1

X1 = np.random.multivariate_normal(means[1], cov, N) # class -1 

X = np.concatenate((X0.T, X1.T), axis = 1) # all data 

y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # labels 
y
X
plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)

plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)

plt.axis('equal')

plt.plot()

plt.show()
# build K

V = np.concatenate((X0.T, -X1.T), axis = 1)

K = matrix(V.T.dot(V)) # see definition of V, K near eq (8)



p = matrix(-np.ones((2*N, 1))) # all-one vector 

# build A, b, G, h 

G = matrix(-np.eye(2*N)) # for all lambda_n >= 0

h = matrix(np.zeros((2*N, 1)))

A = matrix(y) # the equality constrain is actually y^T lambda = 0

b = matrix(np.zeros((1, 1))) 

solvers.options['show_progress'] = False

sol = solvers.qp(K, p, G, h, A, b)



l = np.array(sol['x'])

print('lambda = ')

print(l.T)
epsilon = 1e-6 # just a small number, greater than 1e-9

S = np.where(l > epsilon)[0]



VS = V[:, S]

XS = X[:, S]

yS = y[:, S]

lS = l[S]

# calculate w and b

w = VS.dot(lS)

b = np.mean(yS.T - w.T.dot(XS))



print('w = ', w.T)

print('b = ', b)
# w1*x1 + w2*x2 + b = 0

sepX1 = np.linspace(2.3, 3.3, 100)

sepX2 = -b/w[1] - w[0]*sepX1/w[1]
plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)

plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)

plt.plot(sepX1, sepX2, '-r', label="%f*x1 + %f*x2 + %f = 0"%(w[0], w[1], b))

plt.axis('equal')

plt.plot()

plt.show()
# To support both python 2 and python 3

from __future__ import division, print_function, unicode_literals

# list of points 

import numpy as np 

import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from matplotlib.backends.backend_pdf import PdfPages

np.random.seed(22)



means = [[2, 2], [4, 2]]

cov = [[.7, 0], [0, .7]]

N = 20

X0 = np.random.multivariate_normal(means[0], cov, N) # each row is a data point 

X1 = np.random.multivariate_normal(means[1], cov, N)
#with PdfPages('data.pdf') as pdf:

plt.plot(X0[:, 0], X0[:, 1], 'bs', markersize = 8, alpha = 1)

plt.plot(X1[:, 0], X1[:, 1], 'ro', markersize = 8, alpha = 1)

plt.axis('equal')

plt.ylim(0, 4)

plt.xlim(0, 5)



# hide tikcs 

cur_axes = plt.gca()

cur_axes.axes.get_xaxis().set_ticks([])

cur_axes.axes.get_yaxis().set_ticks([])



plt.xlabel('$x_1$', fontsize = 20)

plt.ylabel('$x_2$', fontsize = 20)

    #pdf.savefig()

    # plt.savefig('logistic_2d.png', bbox_inches='tight', dpi = 300)

plt.show()
X = np.vstack((X0, X1))

y = np.vstack((np.ones((N,1 )), -np.ones((N,1 )))).reshape((2*N,))
C = 100

clf = SVC(kernel = 'linear', C = C)

clf.fit(X, y) 



w_sklearn = clf.coef_.reshape(-1, 1)

b_sklearn = clf.intercept_[0]
print(w_sklearn.T, b_sklearn)
from cvxopt import matrix, solvers

# build K

V = np.concatenate((X0.T, -X1.T), axis = 1)

K = matrix(V.T.dot(V))



p = matrix(-np.ones((2*N, 1)))

# build A, b, G, h 

G = matrix(np.vstack((-np.eye(2*N), np.eye(2*N))))



h = matrix(np.vstack((np.zeros((2*N, 1)), C*np.ones((2*N, 1)))))

A = matrix(y.reshape((-1, 2*N))) 

b = matrix(np.zeros((1, 1))) 

solvers.options['show_progress'] = False

sol = solvers.qp(K, p, G, h, A, b)



l = np.array(sol['x'])

print('lambda = \n', l.T)
l.T
S = np.where(l > 1e-5)[0]

S2 = np.where(l < .99*C)[0]



M = [val for val in S if val in S2] # intersection of two lists
XT = X.T # we need each col is one data point in this alg

VS = V[:, S]

# XS = XT[:, S]

# yS = y[ S]

lS = l[S]

# lM = l[M]

yM = y[M]

XM = XT[:, M]

w_dual = VS.dot(lS).reshape(-1, 1)

b_dual = np.mean(yM.T - w_dual.T.dot(XM))
print(w_dual.T, b_dual) 
def cost(w, lam):

    u = w.T.dot(Z) # as in (23)

    return (np.sum(np.maximum(0, 1 - u)) + \

            .5*lam*np.sum(w*w)) - .5*lam*w[-1]*w[-1]



def grad(w, lam):

    u = w.T.dot(Z) # as in (23)

    H = np.where(u < 1)[1]

    ZS = Z[:, H]

    g = (-np.sum(ZS, axis = 1, keepdims = True) + lam*w)

    g[-1] -= lam*w[-1]

    return g



eps = 1e-6

def num_grad(w):

    g = np.zeros_like(w)

    for i in range(len(w)):

        wp = w.copy()

        wm = w.copy()

        wp[i] += eps 

        wm[i] -= eps 

        g[i] = (cost(wp, lam) - cost(wm, lam))/(2*eps)

    return g 
def grad_descent(w0, eta, lam):

    w = w0

    it = 0 

    while it < 100000:

        it = it + 1

        g = grad(w, lam)

        w -= eta*g

        if (it % 10000) == 1:

            print('iter %d' %it + ' cost: %f' %cost(w, lam))

        if np.linalg.norm(g) < 1e-5:

            break 

    return w 
X0_bar = np.vstack((X0.T, np.ones((1, N)))) # extended data

X1_bar = np.vstack((X1.T, np.ones((1, N)))) # extended data 



Z = np.hstack((X0_bar, - X1_bar)) # as in (22)

lam = 1./C
w0 = np.random.randn(X0_bar.shape[0], 1) 

g1 = grad(w0, lam)

g2 = num_grad(w0)

diff = np.linalg.norm(g1 - g2)

print('Gradient difference: %f' %diff)
w0 = np.random.randn(X0_bar.shape[0], 1) 

w = grad_descent(w0, 0.001, lam)

w_hinge = w[:-1].reshape(-1, 1)

b_hinge = w[-1]

print(w_hinge.T, b_hinge)
def plotResult(X0, X1, w, b, title):

    fig, ax = plt.subplots()



    w0 = w[0]

    w1 = w[1]

    x1 = np.arange(-10, 10, 0.1)

    y1 = -w0/w1*x1 - b/w1

    y2 = -w0/w1*x1 - (b-1)/w1

    y3 = -w0/w1*x1 - (b+1)/w1

    plt.plot(x1, y1, 'k', linewidth = 3)

    plt.plot(x1, y2, 'k')

    plt.plot(x1, y3, 'k')



    # equal axis and lim

    plt.axis('equal')

    plt.ylim(0, 3)

    plt.xlim(2, 4)



    # hide tikcs 

    cur_axes = plt.gca()

    cur_axes.axes.get_xaxis().set_ticks([])

    cur_axes.axes.get_yaxis().set_ticks([])



    # fill two regions

    y4 = 10*x1

    plt.plot(x1, y1, 'k')

    plt.fill_between(x1, y1, color='blue', alpha='0.1')

    plt.fill_between(x1, y1, y4, color = 'red', alpha = '.1')



    plt.xlabel('$x_1$', fontsize=12)

    plt.ylabel('$x_2$', fontsize=12)

    plt.title('Solution found by ' + title, fontsize=12)



    plt.plot(X0[:, 0], X0[:, 1], 'bs', markersize = 8, alpha = .8)

    plt.plot(X1[:, 0], X1[:, 1], 'ro', markersize = 8, alpha = .8)

    plt.show()
plotResult(X0, X1, w_sklearn, b_sklearn, 'sklearn')

plotResult(X0, X1, w_dual, b_dual, 'dual')

plotResult(X0, X1, w_hinge, b_hinge, 'hinge')
# Change C

lsC = [1e-2, 1, 10, 1000]
X0_bar = np.vstack((X0.T, np.ones((1, N)))) # extended data

X1_bar = np.vstack((X1.T, np.ones((1, N)))) # extended data 



Z = np.hstack((X0_bar, - X1_bar)) # as in (22)

for C in lsC:

    lam = 1./C

    w0 = np.random.randn(X0_bar.shape[0], 1) 

    w = grad_descent(w0, 0.001, lam)

    w_hinge = w[:-1].reshape(-1, 1)

    b_hinge = w[-1]

    print(w_hinge.T, b_hinge)

    plotResult(X0, X1, w_hinge, b_hinge, 'hinge')
import numpy as np

import matplotlib.pyplot as plt

from sklearn import svm



# XOR dataset and targets

X = np.c_[(0, 0),

          (1, 1),

          #---

          (1, 0),

          (0, 1)].T

Y = [0] * 2 + [1] * 2
# figure number

fignum = 1



# fit the model

for kernel in ('sigmoid', 'poly', 'rbf'):

    clf = svm.SVC(kernel=kernel, gamma=4, coef0 = 0)

    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane

    fig, ax = plt.subplots()

    plt.figure(fignum, figsize=(4, 3))

    plt.clf()



    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,

                facecolors='None')

    plt.plot(X[:2, 0], X[:2, 1], 'ro', markersize = 8)

    plt.plot(X[2:, 0], X[2:, 1], 'bs', markersize = 8)



    plt.axis('tight')

    x_min, x_max = -2, 3

    y_min, y_max = -2, 3



    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])



    # Put the result into a color plot

    Z = Z.reshape(XX.shape)

    plt.figure(fignum, figsize=(4, 3))

    CS = plt.contourf(XX, YY, np.sign(Z), 200, cmap='jet', alpha = .2)

    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],

                levels=[-.5, 0, .5])

    plt.title(kernel, fontsize = 15)

    plt.xlim(x_min, x_max)

    plt.ylim(y_min, y_max)



    plt.xticks(())

    plt.yticks(())

    fignum = fignum + 1

plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import confusion_matrix 

from sklearn import datasets
## Iris flowers classification using svm

!ls ../input/iris-flower-dataset

filename = "../input/iris-flower-dataset/IRIS.csv"

pdfData = pd.read_csv(filename)

pdfData.head()
pdfData.shape
pdfData.hist(bins=50, figsize=(20,15))

plt.show()
trainSet, testSet = train_test_split(pdfData, test_size=0.2, random_state=42)
lsLabel = set(pdfData["species"])

print(lsLabel)
for l in lsLabel:

    pdfData["label_%s"%l[5:]] = (pdfData["species"] == l)
# Look for correlation

corrMatrix = pdfData.corr()
corrMatrix
for l in lsLabel:

    print(corrMatrix["label_%s"%l[5:]].sort_values(ascending=False))

    print("-"*30)
lsFt = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

print(trainSet[lsFt][trainSet[lsFt].isnull()])

training = trainSet[lsFt].dropna().values

trainingLabel = trainSet["species"].copy()
test = testSet[lsFt].dropna().values

testLabel = testSet["species"].copy()
clf = {}

predictions = {}

lsKernel = ('linear', 'poly', 'rbf')

for kernel in lsKernel:

    clf[kernel] = svm.SVC(kernel=kernel, gamma=4, coef0 = 0)
lsLabel
for kernel in lsKernel:

    print(kernel)

    labels = list(lsLabel)

    clf[kernel].fit(training, trainingLabel)

    

    predictions[kernel] = clf[kernel].predict(test) 



    # model accuracy for X_test   

    accuracy = clf[kernel].score(test, testLabel) 

    print(accuracy)



    # creating a confusion matrix 

    cm = confusion_matrix(testLabel, predictions[kernel], labels) 

    print(cm)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(cm)

    # plt.title('Confusion matrix')

    fig.colorbar(cax)

    ax.set_xticklabels([''] + labels)

    ax.set_yticklabels([''] + labels)

    

    plt.xlabel('Predicted')

    plt.ylabel('True')

    plt.show()

    print("-"*30)
