import numpy as np

import pandas as pd





class Adaline:

    # Adaptive Linear Neurons (Adaline)



    def __init__(self, eta=0.0001, num_epochs=1000):

        self.eta = eta

        self.num_epochs = num_epochs

        self.weights = None

        self._cost = None



    def fit(self, X, y):

        self._cost = []

        X = np.array(X)

        X = np.insert(X, 0, 1, 1)

        self.weights = np.random.normal(loc=0, scale=0.01, size=X.shape[1])

        for _ in range(self.num_epochs):

            net_input = self.net_input(X)

            output = self.activation(net_input)

            errors = y - output

            delta_weights = self.eta * X.transpose().dot(errors)

            self.weights += delta_weights

            self._cost.append((errors ** 2).sum() / 2)



    def net_input(self, X):

        return np.dot(X, self.weights)



    def activation(self, x):

        return x



    def predict(self, x):

        if type(x) == list:

            x = np.array(x)

            x = np.insert(x, 0, 1, 0)

            return np.where(self.activation(self.net_input(x)) >= 0, 1, -1)

        elif (type(x) == pd.DataFrame) or (type(x) == np.ndarray):

            x = np.array(x)

            x = np.insert(x, 0, 1, 1)

            return np.where(self.activation(self.net_input(x)) >= 0, 1, -1)



    def accuracy(self, X_test, y_test):

        return sum(self.predict(X_test) == y_test) / len(y_test)





# if __name__ == "__main__":

    

#     from sklearn.model_selection import train_test_split



#     iris = pd.read_csv("iris.csv")



#     X = iris.iloc[:, :4][:100]

#     y = iris["class"][:100]

#     y = y.apply(lambda c: 1 if c == "Iris-setosa" else -1)



#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)



#     ada = Adaline(eta=0.0001, num_epochs=1000)

#     ada.fit(X_train, y_train)



#     print("Test Accuracy: {:.0%}".format(ada.accuracy(X_test, y_test)))