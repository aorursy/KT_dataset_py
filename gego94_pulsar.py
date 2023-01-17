import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
# import and null values remove

puls = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')

puls.dropna(inplace=True)

puls.head()
# create train and test set

x = puls.drop('target_class', axis=1).values

y = puls.target_class.values

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
# create array of alphas

alphas = [0, 0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000]



res = []

res2 = []

fig, (ax1, ax2, ax3) = plt.subplots(3,2)

axis = [ax1[0], ax1[1], ax2[0], ax2[1], ax3[0], ax3[1]]

plt.rcParams["figure.figsize"] = (30, 20)

# iterate through possible degrees of equation we want to explore

for deg in range(1, 5):

    res.append([])

    res2.append([])

    # iterate through possible alphas of equation we want to explore

    for a in alphas:

        # making ridge model

        clf = make_pipeline(PolynomialFeatures(degree=deg), Ridge(alpha=a))

        # fitting to data

        clf.fit(xTrain, yTrain)

        # calculate score for train and test set

        res[deg-1].append(clf.score(xTest, yTest))

        res2[deg-1].append(clf.score(xTrain, yTrain))

    axis[0].plot(alphas, res[deg-1], label="degree = {}".format(deg))

    axis[1].plot(alphas, res2[deg-1], label="degree = {}".format(deg))

# make the plots

titles = ['Test Set','Train Set','Degree = 1','Degree = 2','Degree = 3','Degree = 4']

for ax in range(0, 6):

    if ax > 1:

        axis[ax].plot(alphas, res[ax-2], label='Test Set')

        axis[ax].plot(alphas, res2[ax-2], label="Train Set")

    axis[ax].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axis[ax].set_title(titles[ax])

plt.tight_layout()
# risults on test data

testSet = pd.DataFrame(res, index=['1','2','3','4'], columns = alphas)

testSet.index.name = 'Degree'

testSet
# risults on train data

trainSet = pd.DataFrame(res2, index=['1','2','3','4'], columns = alphas)

trainSet.index.name = 'Degree'

trainSet
# Questo file contiene uno script per cercare di predire se una stella è un pulsar o meno sulla base di misura da essa derivanti.

# Il dataset è preso da kaggle al seguente indirizzo : https://www.kaggle.com/pavanraj159/predicting-pulsar-star-in-the-universe

# Per tale predizione, si usa la Ridge Regression e l'obiettivo è lo studio di tale regressione al variare dei principali parametri 

# che la interessano.



# come si può vedere, al variare di alfa, non si hanno grandi variazioni nella qualità della predizione, se non un generico calo al 

# drastico aumentare della stessa. Quello che realmente influisce sulla qualità della predizione è il grado del polinomio

# usato nel modello. Infatti, all'aumentare del grado del polinomio, si aumenta l'overfitting, ovvero la capacità del modello di predire i

# dati di training, a discapito dei dati di test. In questo esempio, si ha un miglioramento della predizione nei dati di test fino al 3°

# grado del polinomio, mentre poi si peggiora in maniera incrementale. Al 5° grado, non rappresentato nell'esempio, si ha un peggioramento

# molto elevato nel test set.
