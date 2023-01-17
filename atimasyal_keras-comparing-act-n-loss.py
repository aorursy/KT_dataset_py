from numpy.random import seed

seed(1)

import tensorflow

tensorflow.random.set_seed(1)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras import metrics

import collections

from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve, classification_report
def f(s):

    if s=="negative":

        return 0

    else:

        return 1

def f1(s):

    if s=="M":

        return 2

    else:

        return 1

df=pd.read_csv("../input/project/newthyroid2.csv")

df["Class"]=df.Class.apply(f)

#df["Sex"]=df.Sex.apply(f)

N=df.shape[0]

M=df.shape[1]

x=df.values[:, :M-1]

y=df.values[:, M-1]

#scaler=preprocessing.StandardScaler()

x[0], y[0], x.shape, M
x_train_whole, x_test_whole, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

#scaler.fit(x_train_whole)

#scaler.transform(x_train_whole)

#scaler.transform(x_test_whole)

x.shape
METRICS = ['accuracy']

Activation = ['sigmoid', 'relu', 'elu', 'softmax', 'selu', 'softplus', 'softsign', 'tanh', 'hard_sigmoid', 'exponential', 'linear']

Optimizer = 'adam'

Loss = ['mse', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh', 'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity']

Epochs = 100

rete=list()

retr=list()
df1=pd.DataFrame()

df2=pd.DataFrame()
for act in Activation:

    for los in Loss:

        model = Sequential()

        model.add(Dense(1, activation=act, input_dim=x.shape[1]))

        model.compile(optimizer=Optimizer, loss=los, metrics=METRICS)

        model.fit(x_train_whole, y_train, epochs=Epochs, shuffle=False, validation_data=(x_test_whole, y_test))

        pred1=model.predict_classes(x_test_whole)

        pred2=model.predict_classes(x_train_whole)

        report1 = classification_report(y_test, pred1, output_dict=True)

        df1=df1.append(pd.DataFrame(report1).transpose())

        df1=df1.append(pd.Series(name=act+los))

        

        report2 = classification_report(y_train, pred2, output_dict=True)

        df2=df2.append(pd.DataFrame(report2).transpose())

        rete.append([pred1.sum(), los, act])

        retr.append([pred2.sum(), los, act])
df1.to_csv('file1.csv') 

df2.to_csv('file2.csv') 
y_train.sum(), y_test.sum()
rete
retr
#score=model.evaluate(x_test_whole, y_test)

#score