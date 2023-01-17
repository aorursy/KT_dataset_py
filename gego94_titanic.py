import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import math

import timeit

import statistics
psg = pd.read_csv('../input/TitanicDataset/titanic_data.csv').drop(['Name','Ticket', 'PassengerId'], axis=1)

psg = pd.concat([psg, pd.get_dummies(psg.Sex, drop_first=True).astype(int)], axis=1)

psg = pd.concat([psg, pd.get_dummies(psg.Embarked, drop_first=True).astype(int)], axis=1).rename(columns={'Q':'Embarked_Q','S':'Embarked_S'})

psg.drop(['Sex', 'Embarked'], axis=1, inplace=True)

psg.Cabin = [ 0 if pd.isna(x) else 1 for x in psg.Cabin ]

psg['Age'].fillna((psg['Age'].mean()), inplace=True)

psg
x = psg.drop(['Survived'], axis=1).values

y = psg.Survived.values



from sklearn import svm

from sklearn.metrics import classification_report

C = [0.01, 0.1, 1, 2, 5, 10, 20, 50, 100, 200, 300]

kern = ['rbf', 'sigmoid', 'poly deg: 1', 'poly deg: 2', 'poly deg: 3', 'poly deg: 4', 'poly deg: 5', 'poly deg: 6', 'poly deg: 7']

def trysvm(X_train, X_test, y_train, y_test, C=C, k=kern):

    res = []

    for c in range(len(k)):

        res.append([])

    for c in range(len(C)):

        clf = svm.SVC(kernel='rbf', C=C[c])

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        res[0].append(classification_report(y_test, y_pred, output_dict=True)['accuracy'])

        clf = svm.SVC(kernel='sigmoid', C=C[c])

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        res[1].append(classification_report(y_test, y_pred, output_dict=True)['accuracy'])

        for d in range(1, len(k)-1):

            clf = svm.SVC(kernel='poly', degree=d, C=C[c])

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            res[1+d].append(classification_report(y_test, y_pred, output_dict=True)['accuracy'])

    return res



def disres(res, C=C,k=kern):

    leg=['Std', ' MinMax', 'Norm']

    plt.rcParams["figure.figsize"] = (15, 30)

    fig, ax = plt.subplots(len(k), 1) 

    s = ['C= {}'.format(str(x)) for x in C]

    for a,i in zip(ax,range(len(res[0]))):

        for r in range(len(res)):

            a.plot(s, res[r,i], label=leg[r])

        a.set_title('kernel: %s' %k[i])

        a.tick_params(axis='x', labelrotation=45)

        a.legend()

    plt.tight_layout(h_pad=2)

    plt.show()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_sca = scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x_sca, y, test_size = 0.3, random_state = 1)

res_scale = trysvm(X_train, X_test, y_train, y_test, C)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_sca = scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x_sca, y, test_size = 0.3, random_state = 1)

res_mima = trysvm(X_train, X_test, y_train, y_test, C)
from sklearn.preprocessing import Normalizer

scaler = Normalizer()

x_sca = scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x_sca, y, test_size = 0.3, random_state = 1)

res_norm = trysvm(X_train, X_test, y_train, y_test, C)
disres(np.array([res_scale, res_mima, res_norm]), C)