import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import datasets

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score





def Create_Dataset():



    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return X_train, X_test, y_train, y_test





def Linear_Regression_with_Ordinary_Least_Squares(X_train, X_test, y_train, y_test):



    def variance(X):



        return sum((xi - np.mean(X))**2 for xi in X)



    def covariance(X,y):



        return sum((xi-np.mean(X)) * (yi - np.mean(y)) for xi,yi in zip(X,y))



    def plot(X, y, hypothesis, title):



        plt.figure(figsize=(6, 4))

        plt.style.use("ggplot")

        plt.scatter(X, y, color="green", s=10, label="Actual")

        plt.plot(X, hypothesis, color="black", linewidth=1, label="Prediction")

        plt.xlabel("x")

        plt.ylabel("y")

        plt.title(title, fontdict = {'fontsize' : 10}, fontweight="bold")

        plt.legend()

        plt.show()



    m = covariance(X_train, y_train) / variance(X_train)

    b = np.mean(y_train) - m * np.mean(X_train)



    hypothesis1 = m * X_train + b

    hypothesis2 = m * X_test + b



    print("\nLinear Regression with Ordinary Least Squares")

    print("Coefficient: {}".format(m), "Intercept: {}".format(b))

    plot(X_train, y_train, hypothesis1, "Linear Regression with Ordinary Least Squares (Train Data)")

    plot(X_test, y_test, hypothesis2, "Linear Regression with Ordinary Least Squares (Test Data)")





def Linear_Regression_with_Gradient_Descent(X_train, X_test, y_train, y_test):



    class Simple_Linear_Regression:



        def __init__(self, learning_rate=0.001):



            self.learning_rate = learning_rate

            self.weight = None

            self.bias = None



        def fit(self, X, y):



            self.weight = np.random.randn(X.shape[1])

            self.bias = np.random.randn()



            derivative_weight = 0

            derivative_bias = 0



            for i in range(1000):



                for xi, yi in zip(X, y):



                    hypothesis = np.dot(xi, self.weight) + self.bias

                    derivative_weight += (hypothesis - yi) * xi

                    derivative_bias += (hypothesis - yi)



                derivative_weight /= X.shape[0]

                derivative_bias /= X.shape[0]



                self.weight -= self.learning_rate * derivative_weight

                self.bias -= self.learning_rate * derivative_bias



        def predict(self, X):



            return np.dot(X, self.weight) + self.bias



        def mean_squared_error(self, y, y_prediction):



            return np.mean((y_prediction - y) ** 2)



        def R_Squared(self, y_prediction, y_test):



            SSE = np.sum((y_test - y_prediction) ** 2)

            y_avg = np.sum(y_test) / len(y_test)

            SST = np.sum((y_prediction - y_avg) ** 2)

            RSquared = 1 - (SSE / SST)

            return RSquared



        def plot(self, X, y, y_pred, title):



            plt.figure(figsize=(6, 4))

            plt.scatter(X, y, color="blue", s=10, label= "Actual")

            plt.plot(X, y_pred, color="black", linewidth=1, label = "Prediction")

            plt.xlabel("x")

            plt.ylabel("y")

            plt.title(title, fontdict={'fontsize': 10}, fontweight="bold")

            plt.legend()

            plt.show()



    lr = Simple_Linear_Regression(0.01)

    lr.fit(X_train, y_train)

    prediction_Xtrain = lr.predict(X_train)

    prediction_Xtest = lr.predict(X_test)



    print("\nLinear Regression with Gradient Descent")

    print("Mean Squared Error: {}".format(lr.mean_squared_error(y_test, prediction_Xtest)))

    print("R Squared Value: {}".format(lr.R_Squared(prediction_Xtest, y_test)))

    print("Coefficient: {}".format(lr.weight), "Intercept: {}".format(lr.bias))



    lr.plot(X_train, y_train, prediction_Xtrain, "Linear Regression with Gradient Descent (Train Data)")

    lr.plot(X_test, y_test, prediction_Xtest, "Linear Regression with Gradient Descent (Test Data)")





def Linear_Regression_with_Sklearn(X_train, X_test, y_train, y_test):



    lr = LinearRegression()

    lr.fit(X_train,y_train)

    y_prediction = lr.predict(X_test)

    print("\nLinear Regression with Sklearn")

    print("Mean Squared Error: {}". format(mean_squared_error(y_test, y_prediction)))

    print("R Squared Value: {}".format(r2_score(y_test, y_prediction)))

    print("Coefficient: {}".format(lr.coef_), "Intercept: {}".format(lr.intercept_))





if __name__ == "__main__":



    X_train, X_test, y_train, y_test = Create_Dataset()

    Linear_Regression_with_Ordinary_Least_Squares(X_train, X_test, y_train, y_test)

    Linear_Regression_with_Gradient_Descent(X_train, X_test, y_train, y_test)

    Linear_Regression_with_Sklearn(X_train, X_test, y_train, y_test)
