from keras.layers import Input, Dense, Dropout, Conv1D, Flatten, MaxPooling1D, Concatenate, BatchNormalization, GlobalAveragePooling1D, LeakyReLU

from keras.models import Model, Sequential

from keras import regularizers

from sklearn.model_selection import KFold 

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from keras.callbacks import Callback

from sklearn.linear_model import LogisticRegression, BayesianRidge

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error

from sklearn.metrics import classification_report, accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.manifold import TSNE

import lightgbm as lgb

import math

from sklearn.feature_selection import f_regression

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from scipy import stats

import pandas as pd 

import numpy as np

from sklearn.metrics import roc_curve, auc

import seaborn as sns

sns.set(style="whitegrid")

import matplotlib

from sklearn.naive_bayes import GaussianNB

np.random.seed(203)

import math

from scipy.stats import ks_2samp, normaltest

from sklearn.ensemble import RandomForestRegressor



import os

print(os.listdir("../input"))
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
scaler = StandardScaler()

X = df_train.drop(["target", "ID_code"], axis=1).values

X = scaler.fit_transform(X)

X = X.reshape(200000, 200, 1)

test = df_test.drop(["ID_code"], axis=1).values

test = scaler.transform(test)

test = test.reshape(200000, 200, 1)

y = df_train["target"].values
class roc_callback(Callback):

    def __init__(self,training_data,validation_data):

        self.x = training_data[0]

        self.y = training_data[1]

        self.x_val = validation_data[0]

        self.y_val = validation_data[1]



    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.x)

        roc = roc_auc_score(self.y, y_pred)

        y_pred_val = self.model.predict(self.x_val)

        roc_val = roc_auc_score(self.y_val, y_pred_val)

        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')

        return



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return
def init_model():



    input1 = Input(shape = (200, 1))

    a = Conv1D(22, 2, activation="relu", kernel_initializer="uniform")(input1)

    a = BatchNormalization()(a)

    a = Flatten()(a)

    a = Dropout(0.6)(a)

    a = Dense(50, activation = "relu", kernel_initializer="uniform")(a)

    a = Dropout(0.6)(a)

    output = Dense(1, activation = "sigmoid", kernel_initializer="uniform")(a)

    model = Model(input1, output)

    

    return model
folds = StratifiedKFold(n_splits=12, shuffle=True, random_state=4590)

oof = np.zeros(len(df_train))

test_predictions = np.zeros(len(df_test))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X,y)):

    X_train, X_test = X[trn_idx], X[val_idx]

    y_train, y_test = y[trn_idx], y[val_idx]

    

    model = init_model()

    

    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10, batch_size=512, shuffle=True, callbacks=[roc_callback(training_data=(X_train, y_train),validation_data=(X_test, y_test))])

    oof[val_idx] = model.predict(X_test).reshape(X_test.shape[0],)

    test_predictions+= model.predict(test).reshape(test.shape[0],)/12

roc_auc_score(y, oof)
sub = pd.DataFrame()

sub["ID_code"] = df_test["ID_code"]

sub["target"] = test_predictions

sub.to_csv("submission.csv", index=False)