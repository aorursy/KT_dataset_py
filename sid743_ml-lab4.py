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
from sklearn.datasets import load_digits
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score
dfX = pd.read_csv('/kaggle/input/mnist-original-dataset/train_images_mnist.csv')

dfY = pd.read_csv('/kaggle/input/mnist-original-dataset/train_labels_mnist.csv')
dfX.shape
X = dfX.values

Y = dfY.values
def predict(X,mean_vec,prior):

    preds = np.zeros((X.shape[0],prior.shape[0]))

    for k in range(prior.shape[0]):

        for i in range(X.shape[0]):

            preds[i][k]=-0.5*np.matmul((X[i,:,k]-mean_vec[k,:]),(X[i,:,k]-mean_vec[k,:]).T)+np.log(prior[k]);

    print(preds.shape);

    return np.argmax(preds,axis=1);
def LDA(X,Y,X_test):

    classes = np.unique(Y);

    mean_vec = np.zeros((classes.shape[0],X.shape[1]))

    prior = np.zeros(classes.shape[0])

    cov_mat = np.zeros((X.shape[1],X.shape[1],classes.shape[0]))

    X_testf = np.zeros((X_test.shape[0],X_test.shape[1],classes.shape[0]))

    for k in range(classes.shape[0]):

#         print(np.where(Y==classes[k]))

        X_k = X[np.where(Y==classes[k])[0]][:];

#         print(X_k.shape)

        mean_vec[k][:] = np.mean(X_k,axis=0);

        prior[k] = float(X_k.shape[0])/X.shape[0];

        X_testf[:,:,k] = X_test;

    preds = predict(X_testf,mean_vec,prior)

    for i in range(X_test.shape[0]):

#         print(preds[i])

        preds[i] = classes[preds[i]]

    return preds;
X_train = pd.read_csv('/kaggle/input/mnist-original-dataset/train_images_mnist.csv').values

X_test = pd.read_csv('/kaggle/input/mnist-original-dataset/test_images_mnist.csv').values

Y_train = pd.read_csv('/kaggle/input/mnist-original-dataset/train_labels_mnist.csv').values

Y_test = pd.read_csv('/kaggle/input/mnist-original-dataset/test_labels_mnist.csv').values
print(accuracy_score(LDA(X_train,Y_train,X_test),Y_test))
def QDA(X,Y,X_test):

    classes = np.unique(Y);

    mean_vec = np.zeros((classes.shape[0],X.shape[1]))

    prior = np.zeros(classes.shape[0])

    cov_mat = np.zeros((X.shape[1],X.shape[1],classes.shape[0]))

    funcs = []

    for k in range(classes.shape[0]):

        X_k = X[np.where(Y==k)[0]][:];

#         print(X_k.shape)

        mean_vec[k][:] = np.mean(X_k,axis=0);

        prior[k] = float(X_k.shape[0])/X.shape[0];

        cov_mat[:,:,k] = np.cov(X_k,rowvar = False);

        funcs.append(multivariate_normal(mean = mean_vec[k][:],cov=cov_mat[:,:,k],allow_singular = True))

    preds = np.zeros(X_test.shape[0])

    for i in range(X_test.shape[0]):

        tmp_preds = np.zeros(classes.shape[0])

        for k in range (classes.shape[0]):

            tmp_preds[k]=funcs[k].pdf(X_test[i][:])*prior[k]

        preds[i] = np.argmax(tmp_preds,axis=0)

    

    for i in range(X_test.shape[0]):

#         print(preds[i])

        preds[i] = classes[int(preds[i])]

    return preds.astype(np.int32);
X_train.shape
print(accuracy_score(QDA(X_train,Y_train,X_test),Y_test))
digits = load_digits()

X = digits.data

Y = digits.target
from sklearn.model_selection import train_test_split
X_tr,X_te,Y_tr,Y_te = train_test_split(X, Y, test_size=0.1, random_state=42)
print(accuracy_score(QDA(X_tr,Y_tr,X_te),Y_te))
train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

train.head()
test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

test.head()
X = train.loc[:][train.columns[1:]].values

Y = train.loc[:]['label'].values
Xt= test.loc[:][test.columns[1:]].values

Yt = test.loc[:]['label'].values
X = (X-np.mean(X))/np.std(X)

Xt = (Xt-np.mean(Xt))/np.std(Xt)
print(accuracy_score(QDA(X,Y,Xt),Yt))