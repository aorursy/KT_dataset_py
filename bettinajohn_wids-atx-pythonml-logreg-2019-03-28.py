from sklearn import __version__ as sklearn_version

from distutils.version import LooseVersion



if LooseVersion(sklearn_version) < LooseVersion('0.18'):

    raise ValueError('Please use scikit-learn 0.18 or newer')
from IPython.display import Image

%matplotlib inline

from sklearn import datasets

import numpy as np



iris = datasets.load_iris()

X = iris.data[:, [2, 3]]

y = iris.target



print('Class labels:', np.unique(y))
X.shape
X[0:10,:]
y[0:10]
print(np.sum(y==0))

print(np.sum(y==1))

print(np.sum(y==2))
# Check if the first 50 values in y are from Class 0 (setosa), the next 50 from Class 1 (Versicolor) and the last 50 from Class 2 (Virginica)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=1, stratify=y)
print('Labels counts in y:', np.bincount(y))

print('Labels counts in y_train:', np.bincount(y_train))

print('Labels counts in y_test:', np.bincount(y_test))
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt





def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):



    # setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])



    # plot the decision surface

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),

                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())



    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], 

                    y=X[y == cl, 1],

                    alpha=0.8, 

                    c=colors[idx],

                    marker=markers[idx], 

                    label=cl, 

                    edgecolor='black')



    # highlight test samples

    if test_idx:

        # plot all samples

        X_test, y_test = X[test_idx, :], y[test_idx]



        plt.scatter(X_test[:, 0],

                    X_test[:, 1],

                    c='',

                    edgecolor='black',

                    alpha=1.0,

                    linewidth=1,

                    marker='o',

                    s=100, 

                    label='test set')
X_combined_std = np.vstack((X_train_std, X_test_std))

y_combined = np.hstack((y_train, y_test))
import matplotlib.pyplot as plt

import numpy as np





def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-z))



z = np.arange(-7, 7, 0.1)

phi_z = sigmoid(z)



plt.plot(z, phi_z)

plt.axvline(0.0, color='k')

plt.ylim(-0.1, 1.1)

plt.xlabel('z')

plt.ylabel('$\phi (z)$')



# y axis ticks and gridline

plt.yticks([0.0, 0.5, 1.0])

ax = plt.gca()

ax.yaxis.grid(True)



plt.tight_layout()

#plt.savefig('images/03_02.png', dpi=300)

plt.show()
Image('../input/python-ml-ch03-images/03_03.png',width=700)
Image(filename='../input/regularization/LR-cost.png')
def cost_1(z):

    return - np.log(sigmoid(z))





def cost_0(z):

    return - np.log(1 - sigmoid(z))



z = np.arange(-10, 10, 0.1)

phi_z = sigmoid(z)



c1 = [cost_1(x) for x in z]

plt.plot(phi_z, c1, label='J(w) if y=1')



c0 = [cost_0(x) for x in z]

plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')



plt.ylim(0.0, 5.1)

plt.xlim([0, 1])

plt.xlabel('$\phi$(z)')

plt.ylabel('J(w)')

plt.legend(loc='best')

plt.tight_layout()

#plt.savefig('images/03_04.png', dpi=300)

plt.show()
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(C=100,penalty='l2', random_state=1)

lr.fit(X_train_std, y_train)



plot_decision_regions(X_combined_std, y_combined,

                      classifier=lr, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.tight_layout()

#plt.savefig('images/03_06.png', dpi=300)

plt.show()
#Probability estimate

lr.predict_proba(X_test_std[:3, :])
lr.predict_proba(X_test_std[:3, :]).sum(axis=1)
lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)
lr.predict(X_test_std[:3, :])
lr.predict(X_test_std[0, :].reshape(1, -1))
#Returns the mean accuracy on the given test data and labels.

lr.score(X_test_std, y_test)

lr.score(X_train_std,y_train)
# Find the weights b0 and b1. 

print(lr.coef_)

print(lr.intercept_)
#Image(filename='../input/regularization/04_04.png',width=700)
#Image(filename='../input/regularization/04_06.png',width=700)
Image(filename='../input/regularization/l2-term.png', width=700)
#Image(filename='../input/regularization/04_05.png',width=300)
weights, params = [], []

for c in np.arange(-5, 5):

    lr = LogisticRegression(C=10.**c, random_state=1)

    lr.fit(X_train_std, y_train)

    weights.append(lr.coef_[1])

    params.append(10.**c)



weights = np.array(weights)

plt.plot(params, weights[:, 0],

         label='petal length')

plt.plot(params, weights[:, 1], linestyle='--',

         label='petal width')

plt.ylabel('weight coefficient')

plt.xlabel('C')

plt.legend(loc='upper left')

plt.xscale('log')

#plt.savefig('images/03_08.png', dpi=300)

plt.show()
Image(filename='../input/python-ml-ch03-images/03_09.png', width=700) 
#Image(filename='../input/python-ml-ch03-images/03_10.png', width=600) 
from sklearn.svm import SVC



svm = SVC(kernel='linear', C=0.1, random_state=1)

svm.fit(X_train_std, y_train)



plot_decision_regions(X_combined_std, 

                      y_combined,

                      classifier=svm, 

                      test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.tight_layout()

#plt.savefig('images/03_11.png', dpi=300)

plt.show()
svm.score(X_test_std,y_test)
svm.score(X_train_std,y_train)
from sklearn.linear_model import SGDClassifier



ppn = SGDClassifier(loss='perceptron', n_iter=1000)

lr = SGDClassifier(loss='log', n_iter=1000)

svm = SGDClassifier(loss='hinge', n_iter=1000)
import matplotlib.pyplot as plt

import numpy as np



np.random.seed(1)

X_xor = np.random.randn(200, 2)

y_xor = np.logical_xor(X_xor[:, 0] > 0,

                       X_xor[:, 1] > 0)

y_xor = np.where(y_xor, 1, -1)



plt.scatter(X_xor[y_xor == 1, 0],

            X_xor[y_xor == 1, 1],

            c='b', marker='x',

            label='1')

plt.scatter(X_xor[y_xor == -1, 0],

            X_xor[y_xor == -1, 1],

            c='r',

            marker='s',

            label='-1')



plt.xlim([-3, 3])

plt.ylim([-3, 3])

plt.legend(loc='best')

plt.tight_layout()

#plt.savefig('images/03_12.png', dpi=300)

plt.show()
Image(filename='../input/python-ml-ch03-images/03_13.png', width=700) 
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)

svm.fit(X_xor, y_xor)

plot_decision_regions(X_xor, y_xor,

                      classifier=svm)



plt.legend(loc='upper left')

plt.tight_layout()

#plt.savefig('images/03_14.png', dpi=300)

plt.show()
svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)

svm.fit(X_train_std, y_train)



plot_decision_regions(X_combined_std, y_combined, 

                      classifier=svm, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.tight_layout()

#plt.savefig('images/03_16.png', dpi=300)

plt.show()
from sklearn.svm import SVC



svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)

svm.fit(X_train_std, y_train)



plot_decision_regions(X_combined_std, y_combined,

                      classifier=svm, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.tight_layout()

#plt.savefig('images/03_15.png', dpi=300)

plt.show()
! python ../.convert_notebook_to_script.py --input ch03.ipynb --output ch03.py