import numpy as np

from collections import Counter

from sklearn import datasets

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
class KNN():

    def __init__(self,k=3):

        self.k = k

    def euclidean_distance(self,x1,x2):

        return np.sqrt(np.sum((x1-x2)**2))

    def prediction(self,x):

        distance = [self.euclidean_distance(x,x_train) for x_train in self.X ]

        k_idx = np.argsort(distance)[:self.k]

        k_label = [self.y[i] for i in k_idx]

        common = Counter(k_label).most_common(1)

        return common[0][0]

    def fit(self,X,y):

        self.X = X

        self.y = y

    def predict(self,X):

        pred = [self.prediction(x) for x in X]

        return np.array(pred)

    

iris = datasets.load_iris()

X,y = iris.data,iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)

print(X_train[0])
print(y_train.shape)

print(y_train)
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])
plt.scatter(X[:,0],X[:,1],c=y,cmap = cmap,edgecolors='k')
def accuracy(y_true,y_pred):

    acc = np.sum(y_true == y_pred)/len(y_true)

    return acc
clf = KNN()

clf.fit(X_train,y_train)

predictions = clf.predict(X_test)

print('Accuracy:',accuracy(y_test,predictions))