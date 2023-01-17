import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class GaussianNB:

    def fit(self, X, y):

        self.classes = np.unique(y)

        # separate data points by their classes.
        self.separated = [X[y == i] for i in self.classes]

        # find mean, sigma, and prior probability of each class.
        self.mean = [np.mean(i, axis=0) for i in self.separated]
        self.sigma = [np.var(i, axis=0) for i in self.separated]
        self.priors = [i.shape[0] / X.shape[0] for i in self.separated]

    def predict(self, X):

        post_prob = np.empty([1,3])
        index = []

        # send each data point in X to calculation of probability.
        # calculate probabilities for each class.
        for x in X:

            for i in range(len(self.classes)):

                prior_probability = self.priors[i]
                likelihoods = np.prod(self.likelihood(i,x))
                posterior_probability = prior_probability * likelihoods
                post_prob[0][i] = posterior_probability

            # find the index of highest probability value
            index.append(np.argmax(post_prob))

        # find the class by index
        y_predicted = [self.classes[i] for i in index]

        return y_predicted

    def likelihood(self, index, x):

        mean = self.mean[index]
        sigma = self.sigma[index]
        return (1 / np.sqrt(2 * np.pi * sigma)) * (np.exp(-((x - mean) ** 2 / (2 * sigma))))

    def accuracy(self, y_actual, y_prediction):

        return np.sum(y_actual == y_prediction) / len(y_actual)



iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)
prediction = nb.predict(X_test)

print("Accuracy rate of Gaussian Naive Bayes: ", nb.accuracy(y_test, prediction))


