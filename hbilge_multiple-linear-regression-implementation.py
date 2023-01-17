import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score





def Create_Dataset():



    X, y = datasets.make_regression(n_samples=1000, n_features=4, noise=10, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return X_train, X_test, y_train, y_test





def Linear_Regression_with_Ordinary_Least_Squares(X, y):



    X = np.c_[np.ones(X.shape[0]), X]  # add bias

    beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

    print("\nLinear Regression with Ordinary Least Squares")

    print("Coefficients: {}".format(beta_hat[1:]), "\nIntercept: {}".format(beta_hat[0]))





def Linear_Regression_with_Gradient_Descent(X_train, X_test, y_train, y_test):



    class Multiple_Linear_Regression:



        def __init__(self, learning_rate = 0.001):



            self.learning_rate = learning_rate

            self.weight = None

            self.bias = None



        def fit(self, X, y):



            self.weight = np.random.randn(X.shape[1])

            self.bias = np.random.randn()



            for i in range(10000):



                hypothesis = np.dot(X, self.weight) + self.bias



                derivative_weight = np.mean(X.T * (hypothesis - y), axis = 1)

                derivative_bias = np.mean(hypothesis - y)



                self.weight -= self.learning_rate * derivative_weight

                self.bias -= self.learning_rate * derivative_bias



        def predict(self, X):



            return np.dot(X, self.weight) + self.bias



        def R_Squared(self, y_prediction, y_test):



            SSE = np.sum((y_test - y_prediction) ** 2)

            y_avg = np.sum(y_test) / len(y_test)

            SST = np.sum((y_prediction - y_avg) ** 2)

            RSquared = 1 - (SSE / SST)

            return RSquared



    lr = Multiple_Linear_Regression(0.01)

    lr.fit(X_train, y_train)

    y_prediction = lr.predict(X_test)

    print("\nLinear Regression with Gradient Descent")

    print("Coefficients: {}".format(lr.weight), "\nIntercept: {}".format(lr.bias))

    print("R Squared Value: {}".format(lr.R_Squared(y_prediction, y_test)))





def Linear_Regression_with_Sklearn(X_train, X_test, y_train, y_test):



    lr = LinearRegression()

    lr.fit(X_train, y_train)

    y_prediction = lr.predict(X_test)

    print("\nLinear Regression with Sklearn")

    print("Coefficients: {}".format(lr.coef_), "\nIntercept: {}".format(lr.intercept_))

    print("R Squared Value: {}".format(r2_score(y_test, y_prediction)))





if __name__ == "__main__":



    X_train, X_test, y_train, y_test = Create_Dataset()

    Linear_Regression_with_Ordinary_Least_Squares(X_train, y_train)

    Linear_Regression_with_Gradient_Descent(X_train, X_test, y_train, y_test)

    Linear_Regression_with_Sklearn(X_train, X_test, y_train, y_test)




