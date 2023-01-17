import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import scipy.io
import warnings
import os
print(os.listdir("../input"))
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("Shape of Training data = ", train.shape)
print("Sahpe of Test Data = ", test.shape)
PC = train.iloc[:, 0:1]
train = train.iloc[:, 1:]
print("trainX = ", train.shape, "trainY = ", PC.shape)
variance = np.var(train, axis = 0)>1000
train = train.loc[:, variance]
test = test.loc[:, variance]
print("Shape of Training data = ", train.shape)
print("Sahpe of Test Data = ", test.shape)
from sklearn.model_selection import train_test_split
train_X, val_X, train_Y, val_Y = train_test_split(train, PC, test_size = 0.2, 
                                                 random_state = 0)
print("train_X Shape = ", train_X.shape, "train_Y Shape = ", train_Y.shape)
print("val_X Shape = ", val_X.shape, "val_Y Shape = ", val_Y.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components=0.8)
pca.fit(train_X)
train_X = pca.transform(train_X)
val_X = pca.transform(val_X)
test = pca.transform(test)
print("train_X Shape = ", train_X.shape)
print("val_X Shape = ", val_X.shape)
print("test Shape = ", test.shape)
def normalize(sigma, mean, X):
    X = (X - mean)/sigma
    return X
    
sigma = np.std(train_X, axis = 0)
mean = np.mean(train_X, axis = 0)

train_X = normalize(sigma, mean, train_X)
val_X = normalize(sigma, mean, val_X)
test = normalize(sigma, mean, test)
print(train_X.shape)
train.isna().any().any()
l1Dist = np.array([[3],
                  [1],
                  [2]])
y = np.array([[1],
             [2],
             [3],
             [4]])
ind = np.argsort(l1Dist, axis = 0)
print(ind)
print(y[ind[:2,0]].ravel())
class KNN:
    def __init__(self, k = 5):
        self.k = k
    #X.shape = [1, n]
    def l1(self, X):
        return np.sum(np.abs(X - self.X_train), axis = 1)
    #X.shape = [number of examples, feature] = [m, n]
    #y.shape = [number of examples, 1]
    def fit(self, X, y):
        self.X_train = X
        self.Y_train = y.reshape(y.shape[0], 1)
        
    def predictHelper(self, l1Dist):
        l1Dist = l1Dist.reshape(l1Dist.shape[0], 1)
        l1Arg = np.argsort(l1Dist, axis = 0)[:self.k,0]
        counts = np.bincount(self.Y_train[l1Arg].ravel())
        return np.argmax(counts)
    
    def predict(self, X_test):
        y_test = np.zeros((X_test.shape[0], 1))
        for i in range(X_test.shape[0]):
            l1Dist = self.l1(X_test[i:i+1])
            y_test[i] = self.predictHelper(l1Dist)
        
        return y_test.ravel().astype(np.int)
model = KNN(5)
model.fit(train_X, train_Y.values)
y_test = model.predict(val_X)
boolMap =  (y_test == val_Y.iloc[:,0].values)
acc = np.sum(boolMap)/y_test.shape[0]
print(acc)
sub = pd.DataFrame()
sub['ImageId'] = np.arange(1, test.shape[0]+1)
sub['Label'] = model.predict(test)
sub.to_csv("outKNN.csv", index = False, header = True)
sub.head(n = 10)
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(train_X, train_Y)
print("Training Error = ", neigh.score(train_X, train_Y))
print("Validation Error = ", neigh.score(val_X, val_Y))
(val_Y.values>0).sum()
sub = pd.DataFrame()
sub['ImageId'] = np.arange(1, test.shape[0]+1)
sub['Label'] = neigh.predict(test)
sub.to_csv("outKNN1.csv", index = False, header = True)
sub.head(n = 10)


