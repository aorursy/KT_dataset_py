# Using iris dataset to classify species of flowers using our Scrappy kNN

import random

from scipy.spatial import distance

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
class ScrappyKNN():

    """Sticks to sklearn interface of fit and predict methods

       fit - just initialize the state of kNN with the data

       predict - compute distance to k neighbours and take a vote for majority

       euc - use euclidean distance from scipy  

    """

    def fit(self, X_train, y_train):

        self.X_train = X_train

        self.y_train = y_train



    def predict(self, X_test):

        return [self._kclosest(row, 5) for row in X_test]



    def _kclosest(self, row, k = 1):

        """Finds the eucledian distance between the row (current example) and all the other examples,

           then sort those distances in ascending order, take the k shortest distances, 

           take the accompanying examples, see which classification of those examples is in majority

        """

        train_dists = [[i, self._euc(row, self.X_train[i])] for i in range(0, len(X_train))]

        train_dists.sort(key=lambda x: x[1])



        knns = train_dists[0:k]

        knn_labels = [y_train[knn[0]] for knn in knns]

        return max(set(knn_labels), key=knn_labels.count)



    def _euc(self, x, y):

        return distance.euclidean(x, y)
iris = datasets.load_iris()



X = iris.data

y = iris.target



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)



my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)



predictions = my_classifier.predict(X_test)

print(f"The accuracy of our scrappy k-NN: {accuracy_score(predictions, y_test)}")