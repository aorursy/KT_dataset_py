# Importul bibliotecilor numpy pentru operații matematice, pandas pentru manipularea datelor și matplotlib pentru grafice

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



def load_data(path, header):

    marks_df = pd.read_csv(path, header=header)

    return marks_df





if __name__ == "__main__":

    # Încărcarea datelor din fișierul .txt

    data = load_data("../input/date_regresie_logistica.txt", None)



    # X = valorile caracteristicilor (toate coloanele mai puțin ultima cu rezultatul)

    X = data.iloc[:, :-1]



    # y = valorile rezultat (ultima coloană din dataframe)

    y = data.iloc[:, -1]

    

    theta = np.zeros((X.shape[1], 1))



    # selectarea solicitanților care au fost admiși

    admitted = data.loc[y == 1]



    # selectarea solicitanților care au fost respinși

    not_admitted = data.loc[y == 0]



    # Reprezentarea grafică

    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admiși')

    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Respinși')

    plt.legend()

    plt.show()
X = np.c_[np.ones((X.shape[0], 1)), X]

y = y[:, np.newaxis]

theta = np.zeros((X.shape[1], 1))
from scipy.optimize import fmin_tnc



class LogisticRegressionUsingGD:



    @staticmethod

    # Funcția sigmoid de activare este folosită pentru preluarea fiecărei valori reale între 0 și 1

    def sigmoid(x):

        return 1 / (1 + np.exp(-x))



    @staticmethod

    # Funcția net_input calculează suma ponderată a intrărilor

    def net_input(theta, x):

        return np.dot(x, theta)

    

    # Funcția probility returnează probabilitatea după trecerea prin funcția sigmoid

    def probability(self, theta, x):

        return self.sigmoid(self.net_input(theta, x))

    

    # Calculează funcția de cost pentru toate eșantioanele de antrenare

    def cost_function(self, theta, x, y):

        m = x.shape[0]

        total_cost = -(1 / m) * np.sum(

            y * np.log(self.probability(theta, x)) + (1 - y) * np.log(

                1 - self.probability(theta, x)))

        return total_cost

    

    # Funcția gradient calculează gradientul funcției de cost în punctul teta

    def gradient(self, theta, x, y):

        m = x.shape[0]

        return (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)



    # Antrenează modelul pe setul de date de antrenare

    def fit(self, x, y, theta):

        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient,

                               args=(x, y.flatten()))

        self.w_ = opt_weights[0]

        return self

    

    # Previzionează etichetele claselor

    def predict(self, x):

        theta = self.w_[:, np.newaxis]

        return self.probability(theta, x)

    

    # Calcuează acuratețea modelului

    def accuracy(self, x, actual_classes, probab_threshold=0.5):

        predicted_classes = (self.predict(x) >= probab_threshold).astype(int)

        predicted_classes = predicted_classes.flatten()

        accuracy = np.mean(predicted_classes == actual_classes)

        return accuracy * 100
 # Logistic Regression from scratch using Gradient Descent

model = LogisticRegressionUsingGD()

model.fit(X, y, theta)

accuracy = model.accuracy(X, y.flatten())

parameters = model.w_

print("Acuratețea modelului este {}".format(accuracy))

print("Parametrii modelului folosind metoda coborârii în gradient:")

print(parameters)
x_values = [np.min(X[:, 1] - 2), np.max(X[:, 2] + 2)]

y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]



plt.plot(x_values, y_values, label='Limita deciziei')

plt.xlabel('Note în primul examen')

plt.ylabel('Note în al doilea examen')

plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admiși')

plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Respinși')

plt.legend()

plt.show()
# Antrenarea modelului și stabilirea parametrilor și acurateței sale utilizând biblioteca scikit-learn

model = LogisticRegression()

model.fit(X, y)

parameters = model.coef_

predicted_classes = model.predict(X)

accuracy = accuracy_score(y.flatten(),predicted_classes)

print('Acuratețea modelului folosint scikit-learn este {}'.format(accuracy))

print("Parametrii modelului folosind scikit-learn sunt:")

print(parameters)