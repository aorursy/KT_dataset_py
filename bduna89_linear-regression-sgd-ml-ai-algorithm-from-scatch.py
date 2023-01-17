import numpy as np

import pandas as pd



class LinearRegressionSGD(object):

    

    def __init__(self, eta=0.6, num_epochs=150, random_seed=0):

        self.eta = eta

        self.num_epochs = num_epochs

        np.random.seed(random_seed)

        self.X = None

        self.y = None

        self.X_means = None

        self.y_mean = None

        self.X_std_devs = None

        self.y_std_dev = None

        self.weights = None

        self._mse = []

        

    def fit(self, X, y):

        self.X = X

        self.y = y

        self._standardize()

        min_mse = None

        np.random.seed(0)

        self.X = np.array(self.X)

        self.y = np.array(self.y)

        if len(self.X.shape) == 1:

            self.X = self.X.reshape(-1, 1)

        self.X = np.insert(self.X, 0, 1, 1)

        self.weights = np.random.normal(0, 0.01, self.X.shape[1])

        for epoch in range(self.num_epochs):

            self.X, self.y = self._shuffle()

            for i in range(self.X.shape[0]):

                x = self.X[i]

                for j in range(self.weights.shape[0]):

                    self.weights[j] += -self.eta*(self.predict(x) - self.y[i])*x[j]/len(self.y)

            mse = np.sum((self.predict(self.X) - self.y)**2)/len(self.y)

            if min_mse is None:

                min_mse = mse

                min_weights = self.weights

            else:

                if mse < min_mse:

                    min_mse = mse

                    min_weights = self.weights

            self._mse.append(mse)

        self.weights = min_weights

        self._unstandardize_weights()

        return self

        

    def predict(self, new_data):

        new_data = np.array(new_data)

        if new_data.shape == self.weights.shape:

            return np.dot(new_data, self.weights)

        elif new_data.shape[1] == self.weights.shape[0]:

            return np.dot(new_data, self.weights)

        else:

            new_data = np.insert(new_data, 0, 1, 1)

            return np.dot(new_data, self.weights)

        

    def _shuffle(self):

        rand_perm = np.random.permutation(len(self.y))

        self.X = self.X[rand_perm]

        self.y = self.y[rand_perm]

        return self.X, self.y

    

    def _standardize(self):

        self.X_means = X.mean()

        self.y_mean = y.mean()

        self.X_std_devs = X.std()

        self.y_std_dev = y.std()

        self.X = (self.X-self.X_means)/self.X_std_devs

        self.y = (self.y-self.y_mean)/self.y_std_dev

        

    def _unstandardize_weights(self):

        self.weights[0] = self.y_std_dev*self.weights[0] + self.y_mean

        if self.X.shape[1] > 2:

            for i in range(self.X.shape[1]-1):

                self.weights[0] -= (self.y_std_dev/self.X_std_devs[i])*self.weights[i+1]*self.X_means[i]

            for i in range(1, self.X.shape[1]):

                self.weights[i] = (self.y_std_dev/self.X_std_devs[i-1])*self.weights[i]

        else:

            self.weights[0] -= (self.y_std_dev/self.X_std_devs)*self.weights[1]*self.X_means

            self.weights[1] = (self.y_std_dev/self.X_std_devs)*self.weights[1]

            



# if __name__ == "__main__":

  

#   import matplotlib.pyplot as plt

  

#   data = pd.read_csv("iris.csv")



#   X = data.iloc[:, :3]

#   y = data.iloc[:, 3]

  

#   num_epochs = 150

#   eta = 0.6

#   lr_sgd = LinearRegressionSGD(eta=eta, num_epochs=num_epochs).fit(X, y)

 

#   x_plot = np.arange(1, num_epochs+1)

#   y_plot = lr_sgd._mse

#   plt.plot(x_plot, y_plot)

#   plt.xlabel("Number of Epochs")

#   plt.ylabel("Cost")

#   plt.ylim(0, 0.2)

#   plt.show()

  