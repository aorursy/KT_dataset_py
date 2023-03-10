# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import datasets

import matplotlib.pyplot as plt

import numpy as np, pandas as pd, math

from sklearn.linear_model import LogisticRegression 
np.random.seed(22)



means = [[3.2, 2], [4, 2]]

cov = [[.3, .2], [.2, .3]]

N = 50

N1=10

X0 = np.random.multivariate_normal(means[0], cov, N) # class 1

X1 = np.random.multivariate_normal(means[1], cov, N1) # class -1 

X = np.concatenate((X0.T, X1.T), axis = 1) # all data 

y = np.concatenate((-1*np.ones((1, N)), 1*np.ones((1, N1))), axis = 1) # labels 

# print(X)

# print(y)
plt.figure(figsize=(10,10))
plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)

plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)

plt.plot([3, 3.1], [0, 3.5], lw=1, color="darkorange")



plt.plot([1.18070687, 4.28018474], [0, 3.8], lw=1, color="green") # Run 1

plt.plot([1.35426177, 3.87462372], [0.,  3.8], lw=1, color="darkgreen") # Run 2

plt.axis('equal')

plt.xlabel('x1')

plt.ylabel('x2')

plt.plot()

plt.show()
# Xbar 

X = np.concatenate((np.ones((1, N+N1)), X), axis = 0)
def h(w, x):    

    return np.sign(np.dot(w.T, x))



def has_converged(X, y, w):    

    return np.array_equal(h(w, X), y) 



def perceptron(X, y, w_init):

    w = [w_init]

    eta=0.1

    J=[5]

    N = X.shape[1]

    d = X.shape[0]

    mis_points = []

    n_iteration = 0

    while True:

        # mix data 

        # ho??n v???

        mix_id = np.random.permutation(N)

        for i in range(N):

            xi = X[:, mix_id[i]].reshape(d, 1)

            yi = y[0, mix_id[i]]

            if h(w[-1], xi)[0] != yi: # misclassified point

                mis_points.append(mix_id[i])

                w_new = w[-1] + yi*xi 

                w.append(w_new)

            # tinh J thu i:

                j_new=0            

                for j in range(N):

                    xj = X[:, mix_id[j]].reshape(d, 1)

                    yj = y[0, mix_id[j]]

                    if h(w[-2], xj)[0] != yj:

                        j_new=j_new-yj*np.dot(w[-2].T,xj)[0,0]

                J.append(j_new)

        n_iteration = n_iteration + 1

        if has_converged(X, y, w[-1]) or n_iteration > 1000:

            print("Converged at {} iteration".format(n_iteration))

            break

    return (w, mis_points,J)
d = X.shape[0]

w_init = np.array([[0],[-1],[1]])

print(w_init)

(w,m,J) = perceptron(X, y, w_init)

print(len(w))

# print(J)
print(w[-1])

print(J[-1])

print(w[0])
plt.figure(figsize=(15, 8))



for i in range (100):

    w0 = w[i*50][0]

    w1 = w[i*50][1]

    w2 = w[i*50][2]

    x2 = np.array([0, 3.8])

    x1 = (-w0 - w2*x2)/w1

    plt.plot(x1, x2, lw=1, color="red") # Run 2

    if i<1:

        plt.plot(x1, x2, lw=2, color="blue") # Run 2

plt.plot(x1, x2, lw=3, color="black") # Run 2

plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)

plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)

plt.axis('equal')

plt.xlabel('x1')

plt.ylabel('x2')

plt.plot()

plt.axis([0, 6, 0, 4])

plt.show()
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 

              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])

y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)

def sigmoid(s):

    return 1/(1 + np.exp(-s))



def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):

    w = [w_init]    

    it = 0

    N = X.shape[1]

    d = X.shape[0]

    count = 0

    check_w_after = 20

    while count < max_count:

        # mix data 

        mix_id = np.random.permutation(N)

        for i in mix_id:

            xi = X[:, i].reshape(d, 1)

            yi = y[i]

            zi = sigmoid(np.dot(w[-1].T, xi))

            w_new = w[-1] + eta*(yi - zi)*xi

            count += 1

            # stopping criteria

            if count%check_w_after == 0:                

                if np.linalg.norm(w_new - w[-check_w_after]) < tol:

                    return w

            w.append(w_new)

    return w

eta = .05 

d = X.shape[0]

w_init = np.random.randn(d, 1)



w = logistic_sigmoid_regression(X, y, w_init, eta)

print(w[-1])

print(len(w))
X0 = X[1, np.where(y == 0)][0]

y0 = y[np.where(y == 0)]

X1 = X[1, np.where(y == 1)][0]

y1 = y[np.where(y == 1)]

plt.figure(figsize=(15, 8))



plt.plot(X0, y0, 'ro', markersize = 8)

plt.plot(X1, y1, 'bs', markersize = 8)



xx = np.linspace(-2, 8, 1000)

plt.axis([-2, 8, -0.1, 1.1])

for i in range(100):

    w0 = w[i*30][0][0]

    w1 = w[i*30][1][0]

    threshold = -w0/w1

    yy = sigmoid(w0 + w1*xx)

    plt.plot(xx, yy, 'g-', linewidth = 2)

    if i<4:

        plt.plot(xx, yy, 'b-', linewidth = 2)

plt.plot(xx, yy,'*', 'black', linewidth = 2)

plt.plot(threshold, .5, 'y^', markersize = 2)

plt.xlabel('studying hours')

plt.ylabel('predicted probability of pass')

plt.grid()

plt.show()
pdfBreast = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
pdfBreast.shape
pdfBreast.head()
pdfBreast.describe()
pdfBreast.columns
lsCol = pdfBreast.columns

ftCol = [c for c in lsCol if c not in ["id", "diagnosis", "Unnamed: 32"]]

lbCol = "diagnosis"
ftCol
data = pdfBreast[ftCol].values

label = (pdfBreast[lbCol]=='M').values

# print(data)

# print(label)
# Area Mean vs Label

tumorSize = pdfBreast["radius_mean"].values

# print(tumorSize)

# print(label)
plt.figure(figsize=(10, 5))

plt.plot(tumorSize, label, 'bo')

# plt.axis([140, 190, 45, 75])

plt.xlabel('Tumor Size')

plt.ylabel('Malignant')

plt.grid(True)

plt.show()
# TODO:

logReg = LogisticRegression()

logReg.fit(tumorSize.reshape(-1, 1), label)
X_new = np.linspace(0, 30, 100).reshape(-1, 1)

y_proba = logReg.predict_proba(X_new)

plt.plot(X_new, y_proba[:, 1], "g-", label="Predicting")

plt.plot(tumorSize, label, 'bo')

# plt.axis([140, 190, 45, 75])

plt.xlabel('Tumor Size')

plt.ylabel('Malignant')

plt.grid(True)

plt.show()
pdfTitanic = pd.read_csv("/kaggle/input/titanic/train_and_test2.csv")
pdfTitanic.shape
pdfTitanic.head()
pdfTitanic.describe()
# pdfTitanic.hist(bins=50, figsize=(20,15))

# plt.show()
pdfTitanic.plot(kind="scatter", x="Age", y="Fare", alpha=0.1)
pdfTitanic.plot(

    kind="scatter", x="Age", y="Fare", alpha=0.4,figsize=(10,7),

    c="2urvived", cmap=plt.get_cmap("jet"), colorbar=True,

)

plt.legend()
lsCol = [c for c in pdfTitanic.columns if "zero" not in c]
# TODO:a

pdfData = pdfTitanic[lsCol]

lbCol = "2urvived"

data = pdfTitanic[lsCol].values

label = (pdfTitanic[lbCol]==1).values

X = pdfTitanic[["Age","Sex","Embarked","Pclass"]].values

print(X.shape)

print(pdfTitanic[lbCol].shape)

plt.figure(figsize=(10, 5))

plt.plot(X[:,0].reshape(-1,1), label, 'bo')

# plt.axis([140, 190, 45, 75])

plt.xlabel('age')

plt.ylabel('dead/alive')

plt.grid(True)

plt.show()
print(X.shape)

print(label.shape)

fromNumber=100

number=400

result=logReg.fit(X[fromNumber:number,:], label[fromNumber:number])
more=200

y_pred = logReg.predict(X[number:number+more,:])

print('Accuracy of logistic regression classifier on test set: {:.3f}'.format(logReg.score(X[number:number+more,:], label[number:number+more])))

print(X.shape)

print(y.shape)
logReg.predict_proba(X[number:number+20,:])
# X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 

#               2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])

# y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

Xb = np.array(X)



Xb=Xb.T

Xb = np.concatenate((np.ones((1,Xb.shape[1])), Xb), axis = 0)

print(Xb.shape)

y=np.array(pdfTitanic[lbCol])

print(Xb.shape)

print(y.shape)

def sigmoid(s):

    return 1/(1 + np.exp(-s))

# tang max_count se bi error

def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 500):

    w = [w_init]    

    it = 0

    N = X.shape[1]

    d = X.shape[0]

    count = 0

    check_w_after = 20

    while count < max_count:

        # mix data 

        # tang mix_data la bi error

        mix_id = np.random.permutation(50)

        for i in mix_id:

            xi = X[:, i].reshape(d, 1)

            yi = y[i]

            zi = sigmoid(np.dot(w[-1].T, xi))

#             print("x:",xi)

#             print("y:",yi)

#             print("z: ",zi)

            w_new = w[-1] + eta*(yi - zi[0][0])*xi

            count += 1

#             print(w_new)

            # stopping criteria

            if count%check_w_after == 0:                

                if np.linalg.norm(w_new - w[-check_w_after]) < tol:

                    return w

            w.append(w_new)

        return w

eta = .0005 

d = Xb.shape[0]

w_init = np.random.randn(d, 1)

print(Xb.shape)

print(y.shape)

w = logistic_sigmoid_regression(Xb, y, w_init, eta)

print(w[-1])

print(X.shape)

print(y.shape)

print(w[-1])