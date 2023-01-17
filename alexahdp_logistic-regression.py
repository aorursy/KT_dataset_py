import numpy as np

import pandas as pd

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

import path

from scipy.optimize import fmin_tnc
df = pd.read_csv('../input/train.csv')



y = df['Survived']

x = df.drop('Survived', axis=1)



# эти поля не используем

x = x.drop(['Cabin', 'Name', 'Ticket'], axis=1)



# В колонке Age много пропущенных значений. Заменим их средними

x['Age'] = x['Age'].mean()



# в поле Embarked есть пропущенные значения

# заменим их модой

x['Embarked'] = x['Embarked'].fillna(x['Embarked'].mode()[0])



# посольку Sex - категориальный признак, заменим эту колонку на числа

x['Sex'] = (x['Sex'] == 'male').astype('int')

x['Sex'].value_counts()



x = pd.get_dummies(x)



x['Pclass'] = x['Pclass'].astype('category')



x = pd.get_dummies(x)



X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.25, random_state=42)



lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_valid)

accuracy_score(y_valid, y_pred)
class LogisticRegressionUsingGD:



    @staticmethod

    def sigmoid(x):

        return 1 / (1 + np.exp(-x))



    def probability(self, w, x):

        return self.sigmoid(np.dot(x, w))



    def cost_function(self, w, x, y):

        m = x.shape[0]

        A = self.probability(w, x)

        total_cost = -(1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

        return total_cost



    def gradient(self, w, x, y):

        m = x.shape[0]

        return (1 / m) * np.dot(x.T, self.sigmoid(np.dot(x, w)) - y)



    def fit(self, x, y, w):

        opt_weights = fmin_tnc(

            func=self.cost_function,

            x0=w,

            fprime=self.gradient,

            args=(x, y.flatten())

        )

        self.w_ = opt_weights[0]

        return self



    def predict(self, x):

        theta = self.w_[:, np.newaxis]

        predicted = self.probability(theta, x)

        ss = list(map(lambda v: 1 if v > 0.5 else 0, predicted))

        return np.array(ss)



    def accuracy(self, x, actual_classes):

        return accuracy_score(actual_classes, x)



model = LogisticRegressionUsingGD()



y_train_ = np.array([y_train.values])

theta = np.zeros((X_train.shape[1], 1))



model.fit(X_train, y_train_, theta)

y_predicted = model.predict(X_valid)



print("The accuracy of the model is {}".format(model.accuracy(y_valid, y_predicted)))