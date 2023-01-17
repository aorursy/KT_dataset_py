# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import datetime

import time

import warnings

warnings.filterwarnings('ignore')





from scipy.stats import norm

from scipy import stats

from sklearn import preprocessing



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

sns.set_style('whitegrid')

%matplotlib inline



#VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

from patsy import dmatrices



#Modelling

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier

from sklearn.neural_network import MLPClassifier

import xgboost



#Neural Nets

from keras.models import Sequential

from keras.layers import Dense

from keras.preprocessing import text, sequence

from keras import layers, models, optimizers

# Evaluation metrics

from sklearn import metrics 

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
def printContingencyTable(y_cv, Y_pred):

    confusion_matrix = metrics.confusion_matrix(y_cv, Y_pred)

    plt.matshow(confusion_matrix)

    plt.title('Confusion matrix')

    plt.colorbar()

    plt.ylabel('Churned')

    plt.xlabel('Predicted')

    plt.show()

    print("precision_score : ", metrics.precision_score(y_cv, Y_pred))

    print("recall_score : ", metrics.recall_score(y_cv, Y_pred))

    print("f1_score : ", metrics.f1_score(y_cv, Y_pred))

    print(confusion_matrix)
col=["target","ID_code"]

X = train.drop(col,axis=1)

y = train["target"]



X_test  = test.drop("ID_code",axis=1)



train_x, cv_x, train_y, cv_y = train_test_split(X, y, test_size=0.30, random_state=42)
logreg = LogisticRegression()

logreg.fit(train_x, train_y)

Y_pred = logreg.predict(cv_x)
printContingencyTable(cv_y, Y_pred)
DT = DecisionTreeClassifier()

DT.fit(train_x, train_y)

Y_pred = DT.predict(cv_x)

printContingencyTable(cv_y, Y_pred)