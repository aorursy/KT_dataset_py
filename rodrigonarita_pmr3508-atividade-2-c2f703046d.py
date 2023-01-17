import pandas as pd
from pandas import DataFrame 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import naive_bayes
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import fbeta_score
train_data = pd.read_csv("../input/spam-detection/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
test_data = pd.read_csv("../input/spam-detection/test_features.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train_data.describe()
correlation=train_data.corr()
correlation.head()
correlation_less=correlation[abs(correlation.ham) < 0.05].sort_values(by=['ham'])
correlation_less
train_features=train_data.drop(['ham','Id',
                               'word_freq_report', 'word_freq_will', 'Id', 'word_freq_address', 'word_freq_parts']
                               , axis=1)
x_train = train_features
y_train = train_data.ham
x_train.shape
y_train
#y_train=y_train.astype(float)
scores=[]
classifier = naive_bayes.GaussianNB()
scores = cross_val_score(classifier, x_train, y_train, cv=10, scoring='f1')
scores.mean()
train_features=train_data.drop(['ham','Id',
                               'word_freq_report', 'word_freq_will', 'Id', 'word_freq_address', 'word_freq_parts']
                               , axis=1)
classifier.fit(x_train,y_train)
ntest = test_data.drop(["Id", 'word_freq_report', 'word_freq_will', 'Id', 'word_freq_address', 'word_freq_parts'], axis = 1)
Ytest = classifier.predict(ntest)
prediction = pd.DataFrame(index = test_data.index)
prediction['ham'] = Ytest
prediction
prediction.to_csv('prediction.csv')