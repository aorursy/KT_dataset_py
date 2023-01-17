# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def asRowMatrix(X):

    if len(X) == 0:

        return np.array([])

    mat = np.empty((0, X[0].size), dtype=X[0].dtype)

    for row in X:

        mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))

    return mat



def pca(X, y, num_components=0):

    [n,d] = X.shape

    if (num_components <= 0) or (num_components>n):

        num_components = n

    mu = X.mean(axis=0)

    X = X - mu

    if n>d:

        C = np.dot(X.T,X)

        [eigenvalues,eigenvectors] = np.linalg.eigh(C)

    else:

        C = np.dot(X,X.T)

        [eigenvalues,eigenvectors] = np.linalg.eigh(C)

        eigenvectors = np.dot(X.T,eigenvectors)

        for i in range(n):

            eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])

    # or simply perform an economy size decomposition

    # eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)

    # sort eigenvectors descending by their eigenvalue

    idx = np.argsort(-eigenvalues)

    eigenvalues = eigenvalues[idx]

    eigenvectors = eigenvectors[:,idx]

    # select only num_components

    eigenvalues = eigenvalues[0:num_components].copy()

    eigenvectors = eigenvectors[:,0:num_components].copy()

    return [eigenvalues, eigenvectors, mu]



def lda(X, y, num_components=0):

    print(X.shape)

    y = np.asarray(y)

    [n,d] = X.shape

    c = np.unique(y)

    if (num_components <= 0) or (num_component>(len(c)-1)):

        num_components = (len(c)-1)

    meanTotal = X.mean(axis=0)

    Sw = np.zeros((d, d), dtype=np.float32)

    Sb = np.zeros((d, d), dtype=np.float32)

    for i in c:

        Xi = X[np.where(y==i)[0],:]

        meanClass = Xi.mean(axis=0)

        Sw = Sw + np.dot((Xi-meanClass).T, (Xi-meanClass))

        Sb = Sb + n * np.dot((meanClass - meanTotal).T, (meanClass - meanTotal))

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)

    idx = np.argsort(-eigenvalues.real)

    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:,idx]

    eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)

    eigenvectors = np.array(eigenvectors[0:,0:num_components].real, dtype=np.float32, copy=True)

    return [eigenvalues, eigenvectors]



def fisherfaces(X,y,num_components=0):

    y = np.asarray(y)

    [n,d] = X.shape



    c = len(np.unique(y))

    [eigenvalues_pca, eigenvectors_pca, mu_pca] = pca(X, y, (n-c))

    [eigenvalues_lda, eigenvectors_lda] = lda(project(eigenvectors_pca, X, mu_pca), y, num_components)

    eigenvectors = np.dot(eigenvectors_pca,eigenvectors_lda)

    return [eigenvalues_lda, eigenvectors, mu_pca]



def project(W, X, mu=None):

    if mu is None:

        return np.dot(X,W)

    return np.dot(X - mu, W)

def reconstruct(W, Y, mu=None):

    if mu is None:

        return np.dot(Y,W.T)

    return np.dot(Y, W.T) + mu

class AbstractDistance(object):

    def __init__(self, name):

        self._name = name



    def __call__(self,p,q):

        raise NotImplementedError("Every AbstractDistance must implement the __call__ method.")



    @property

    def name(self):

        return self._name



    def __repr__(self):

        return self._name



class EuclideanDistance(AbstractDistance):



    def __init__(self):

        AbstractDistance.__init__(self,"EuclideanDistance")



    def __call__(self, p, q):

        p = np.asarray(p).flatten()

        q = np.asarray(q).flatten()

        return np.sqrt(np.sum(np.power((p-q),2)))



class CosineDistance(AbstractDistance):



    def __init__(self):

        AbstractDistance.__init__(self,"CosineDistance")



    def __call__(self, p, q):

        p = np.asarray(p).flatten()

        q = np.asarray(q).flatten()

        return -np.dot(p.T,q) / (np.sqrt(np.dot(p,p.T)*np.dot(q,q.T)))



class BaseModel(object):

    def __init__(self, X=None, y=None, dist_metric=EuclideanDistance(), num_components=0):

        self.dist_metric = dist_metric

        self.num_components = 0

        self.projections = []

        self.W = []

        self.mu = []

        if (X is not None) and (y is not None):

            self.compute(X,y)



    def compute(self, X, y):

        raise NotImplementedError("Every BaseModel must implement the compute method.")



    def predict(self, X):

        minDist = np.finfo('float').max

        minClass = -1

        Q = project(self.W, X.reshape(1,-1), self.mu)

        for i in range(len(self.projections)):

            dist = self.dist_metric(self.projections[i], Q)

            if dist < minDist:

                minDist = dist

                minClass = self.y[i]

        return minClass



class EigenfacesModel(BaseModel):



    def __init__(self, X=None, y=None, dist_metric=EuclideanDistance(), num_components=0):

        super(EigenfacesModel, self).__init__(X=X,y=y,dist_metric=dist_metric,num_components=num_components)



    def compute(self, X, y):

        [D, self.W, self.mu] = pca(asRowMatrix(X),y, self.num_components)

    # store labels

        self.y = y

    # store projections

        for xi in X:

            self.projections.append(project(self.W, xi.reshape(1,-1), self.mu))



class FisherfacesModel(BaseModel):



    def __init__(self, X=None, y=None, dist_metric=EuclideanDistance(), num_components=0):

        super(FisherfacesModel, self).__init__(X=X,y=y,dist_metric=dist_metric,num_components=num_components)



    def compute(self, X, y):

        [D, self.W, self.mu] = fisherfaces(asRowMatrix(X),y, self.num_components)

        # store labels

        self.y = y

        # store projections

        for xi in X:

            self.projections.append(project(self.W, xi.reshape(1,-1), self.mu))





def normalize(X, low, high, dtype=None):

    X = np.asarray(X)

    minX, maxX = np.min(X), np.max(X)

    # normalize to [0...1].	

    X = X - float(minX)

    X = X / float((maxX - minX))

    # scale to [low...high].

    X = X * (high-low)

    X = X + low

    if dtype is None:

        return np.asarray(X)

    return np.asarray(X, dtype=dtype)







# read images( from the folder containing yalefaces data )

# print(X)





from sklearn.model_selection import train_test_split

X = np.load("/kaggle/input/olivetti/olivetti_faces.npy")

y = np.load("/kaggle/input/olivetti/olivetti_faces_target.npy")

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

print(X.shape,y.shape)
# compute the eigenfaces model

model = FisherfacesModel (X_train, y_train,CosineDistance() )

# get a prediction for the first observation

print (" expected " , y [22] , "/" , " predicted =" , model . predict ( X [22]))

y_pred =np.random.random(y_test.shape)

for i in range(len(X_test)):

    y_pred[i] = model.predict(X_test[i])
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score, classification_report
print(classification_report(y_test, y_pred))
print('Accuracy ' + str(accuracy_score(y_test, y_pred)))

print('f1_score ' + str(f1_score(y_test, y_pred,average='micro')))

print('Recall score ' + str(recall_score(y_test, y_pred,average='micro')))

print('Precision ' + str(precision_score(y_test, y_pred,average='micro')))