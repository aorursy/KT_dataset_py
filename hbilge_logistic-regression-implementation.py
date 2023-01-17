import numpy as np

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score





def Create_Dataset():



    X, y = datasets.make_classification(n_samples=10000, n_features=4, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    return X_train, X_test, y_train, y_test





def Binary_Logistic_Regression(X_train, X_test, y_train, y_test):



    class Logistic_Regression:



        def __init__(self, learning_rate=0.001, n_iteration=10000):



            self.learning_rate = learning_rate

            self.n_iteration = n_iteration

            self.weight = None

            self.bias = None



        def fit(self, X, y):



            self.weight = np.zeros(X.shape[1])

            self.bias = 0



            for i in range(self.n_iteration):



                linear_equation = np.dot(X, self.weight) + self.bias

                hypothesis = self.sigmoid_function(linear_equation)



                #cost_function = (-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)).mean()



                derivative_weight = np.mean(X.T * (hypothesis - y), axis=1)

                derivative_bias = np.mean(hypothesis - y)



                self.weight -= self.learning_rate * derivative_weight

                self.bias -= self.learning_rate * derivative_bias



        def predict(self, X):



            linear_equation = np.dot(X, self.weight) + self.bias

            hypothesis = self.sigmoid_function(linear_equation)

            result = [1 if i >= 0.5 else 0 for i in hypothesis]

            return result



        def sigmoid_function(self, s):



            return 1 / (1 + (np.exp(-s)))



        def accuracy_score(self, y, y_prediction):



            return np.sum(y == y_prediction) / len(y)





    lr = Logistic_Regression(0.01, 15000)

    lr.fit(X_train, y_train)

    y_prediction = lr.predict(X_test)

    print("\nAccuracy score of Python Implementation: {}".format(lr.accuracy_score(y_test, y_prediction)))





def Logistic_Regression_with_Sklearn(X_train, X_test, y_train, y_test):



    lr = LogisticRegression()

    lr.fit(X_train, y_train)

    y_prediction = lr.predict(X_test)

    print("Accuracy score of Sklearn: {}".format(accuracy_score(y_test, y_prediction)))





if __name__ == "__main__":



    X_train, X_test, y_train, y_test = Create_Dataset()

    Binary_Logistic_Regression(X_train, X_test, y_train, y_test)

    Logistic_Regression_with_Sklearn(X_train, X_test, y_train, y_test)