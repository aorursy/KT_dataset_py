import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path
# Lets go ahead  and have a look at data

DATA_PATH = "../input/santander-customer-transaction-prediction/"  



train = pd.read_csv(str(Path(DATA_PATH) / "train.csv"))

test = pd.read_csv(str(Path(DATA_PATH) / "test.csv"))



print("Train and test shapes", train.shape, test.shape)
train.columns, test.columns
train.target.value_counts()
# https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

def woe(X, y):

    tmp = pd.DataFrame()

    tmp["variable"] = X

    tmp["target"] = y

    var_counts = tmp.groupby("variable")["target"].count()

    var_events = tmp.groupby("variable")["target"].sum()

    var_nonevents = var_counts - var_events

    tmp["var_counts"] = tmp.variable.map(var_counts)

    tmp["var_events"] = tmp.variable.map(var_events)

    tmp["var_nonevents"] = tmp.variable.map(var_nonevents)

    events = sum(tmp["target"] == 1)

    nonevents = sum(tmp["target"] == 0)

    tmp["woe"] = np.log(((tmp["var_nonevents"])/nonevents)/((tmp["var_events"])/events))

    tmp["woe"] = tmp["woe"].replace(np.inf, 0).replace(-np.inf, 0)

    tmp["iv"] = (tmp["var_nonevents"]/nonevents - tmp["var_events"]/events) * tmp["woe"]

    iv = tmp.groupby("variable")["iv"].last().sum()

    return tmp["woe"], tmp["iv"], iv
iv_values = []

feats = ["var_{}".format(i) for i in range(200)]

y = train["target"]

for f in feats:

    X = pd.qcut(train[f], 10, duplicates='drop')

    _, _, iv = woe(X, y)

    iv_values.append(iv)

    

iv_inds = np.argsort(iv_values)[::-1][:50]

iv_values = np.array(iv_values)[iv_inds]

feats = np.array(feats)[iv_inds]

plt.figure(figsize=(10, 16))

sns.barplot(y=feats, x=iv_values, orient='h')

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler
feats = ["var_{}".format(i) for i in range(200)]

X = train[feats]

X_test = test[feats]

y = train["target"]



cvlist = list(StratifiedKFold(5, random_state=12345786).split(X, y))

scaler = StandardScaler()



X_sc = scaler.fit_transform(X)

X_test_sc = scaler.fit_transform(X_test)



lr = LogisticRegression()

y_preds_lr = cross_val_predict(lr, X_sc, y, cv=cvlist, method="predict_proba")[:, 1]



lr.fit(X_sc, y)

y_test_preds_lr = lr.predict_proba(X_test_sc)[:, 1] 

roc_auc_score(y, y_preds_lr)
sns.distplot(y_preds_lr)

sns.distplot(y_test_preds_lr)

plt.show()
import lightgbm as lgb

#model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.1, num_leaves=2, subsample=0.4, colsample_bytree=0.4)



#y_preds_lgb = np.zeros((len(y)))

#for i, (tr_idx, val_idx) in enumerate(cvlist):

#    X_dev, y_dev = X.iloc[tr_idx], y.iloc[tr_idx]

#    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

#    model.fit(X_dev, y_dev, eval_set=[(X_val, y_val)], eval_metric="auc", verbose=50, early_stopping_rounds=200)

#    val_preds = model.predict_proba(X_val)[:, 1]

#    y_preds_lgb[val_idx] = val_preds

#    print("Score for fold {} is {}".format(i, roc_auc_score(y_val, val_preds)))

    

#print("Overall Score for oof predictions ", roc_auc_score(y, y_preds_lgb))
#model = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.1, num_leaves=8, subsample=0.6, colsample_bytree=0.6)

#model.fit(X, y)

#y_test_preds_lgb = model.predict_proba(X_test)[:, 1]

#sns.distplot(y_preds)

#sns.distplot(y_test_preds_lgb)
from scipy.stats import gmean
np.mean([0.9, 0.9, 0.9, 0.98, 0.9])
gmean([0.9, 0.9, 0.9, 0.98, 0.9])
!pip install -U lightgbm
import lightgbm as lgb

model = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=200000, learning_rate=0.02, num_leaves=2, subsample=0.4, colsample_bytree=0.4, seed=1)



y_preds_lgb = np.zeros((len(y)))

test_preds_allfolds = []

for i, (tr_idx, val_idx) in enumerate(cvlist):

    X_dev, y_dev = X.iloc[tr_idx], y.iloc[tr_idx]

    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model.fit(X_dev, y_dev, eval_set=[(X_val, y_val)], eval_metric="auc", verbose=5000, early_stopping_rounds=1000)

    val_preds = model.predict_proba(X_val)[:, 1]

    test_preds = model.predict_proba(X_test)[:, 1]

    test_preds_allfolds.append(test_preds)

    y_preds_lgb[val_idx] = val_preds

    print("Score for fold {} is {}".format(i, roc_auc_score(y_val, val_preds)))

    # break

print("Overall Score for oof predictions ", roc_auc_score(y, y_preds_lgb))
y_test_preds_lgb = gmean(test_preds_allfolds, 0)

sns.distplot(y_preds_lgb)

sns.distplot(y_test_preds_lgb)
sub = test[["ID_code"]]

sub["target"] = y_test_preds_lgb

sub.to_csv("submission_lgbm2_v1.csv", index=False)
weighted_preds = y_preds_lr* 0.05 + y_preds_lgb * 0.95

weighted_test_preds = y_test_preds_lr* 0.05 + y_test_preds_lgb * 0.95

roc_auc_score(y, weighted_preds)
public_sub = pd.read_csv("../input/santander-lgb-new-features-rank-mean-10-folds/submission_LGBM.csv")

public_sub.head()
sub["target"] = weighted_test_preds
sub["target"] = 0.2*sub["target"].rank() + 0.8*public_sub["target"]

sub.to_csv("submission_blend.csv", index=False)
import keras

from keras.layers import (Flatten, Conv1D, Conv2D, Input, Dense, Dropout, BatchNormalization,

                          concatenate, GaussianNoise, Reshape, TimeDistributed, LeakyReLU, PReLU, Embedding)

from keras.models import Model, load_model, save_model

from keras.optimizers import SGD, Adam

from sklearn.base import BaseEstimator, ClassifierMixin

from pathlib import Path

from keras.callbacks import Callback



class ROC_AUC(Callback):

    def __init__(self, validation_data):

        self.X_val, self.y_val = validation_data

    

    def on_epoch_end(self, epoch, logs={}):

        print("ROC AUC for this fold, is ", roc_auc_score(self.y_val, self.model.predict(X_val)))

        

class NNv1(BaseEstimator, ClassifierMixin):

    def __init__(self,

                 inp_shape=200,

                 gaussian_noise=0.01,

                 dense1_dim=32,

                 dense2_dim=32,

                 dense1_kwargs=None,

                 dense2_kwargs=None,

                 classifier_kwargs=None,

                 optimizer=SGD,

                 opt_kwargs=None,

                 ):

        self.inp_shape = inp_shape

        self.gaussian_noise = gaussian_noise

        self.dense1_dim = dense1_dim

        self.dense2_dim = dense2_dim

        self.dense1_kwargs = dense1_kwargs

        self.dense2_kwargs = dense2_kwargs

        self.classifier_kwargs = classifier_kwargs

        self.optimizer = optimizer

        self.opt_kwargs = opt_kwargs

        self._default_initiaization()



    def _default_initiaization(self):

        if self.dense1_kwargs is None:

            self.dense1_kwargs = {"kernel_initializer": "glorot_uniform"}

        if self.dense2_kwargs is None:

            self.dense2_kwargs = {"kernel_initializer": "he_uniform"}

        if self.classifier_kwargs is None:

            self.classifier_kwargs = {"kernel_initializer": "he_uniform"}

        if self.opt_kwargs is None:

            self.opt_kwargs = {}



    def _build_model(self):

        inp = Input(shape=(self.inp_shape,))

        # x = GaussianNoise(self.gaussian_noise)(inp)

        x = Reshape((self.inp_shape, 1))(inp)

        d1 = Dense(self.dense1_dim, activation='tanh',)(x)

        #d1 = TimeDistributed(Dropout(0.2))(d1)

        d2 = Dense(self.dense1_dim, activation='relu',)(x)

        #d2 = PReLU()(d2)

        #d2 = TimeDistributed(Dropout(0.2))(d2)

        x = concatenate([d1, d2])

        x = Flatten()(x)

        out = Dense(1, activation='sigmoid', **self.classifier_kwargs)(x)



        model = Model(inputs=inp, outputs=out)

        opt = self.optimizer(**self.opt_kwargs)

        model.compile(loss='binary_crossentropy', optimizer=opt)

        return model



    def fit(self, X, y, *args, **kwargs):

        self.model = self._build_model()

        print(self.model.summary())

        self.model.fit(X, y, *args, **kwargs)

        return self



    def predict(self, X, y=None, weight_path=None, **kwargs):

        if self.model:

            if weight_path is not None:

                self.model.load_weights(weight_path)

            y_hat = self.model.predict(X, **kwargs)

        else:

            raise ValueError("Model not fit yet")

        return y_hat

model = NNv1(opt_kwargs = {"lr": 0.01, "momentum": 0.9, "nesterov": True, "clipnorm": 1})

y_preds_nn = np.zeros((len(y)))

for tr_idx, val_idx in cvlist:

    X_dev, y_dev = X_sc[tr_idx], y.iloc[tr_idx]

    X_val, y_val = X_sc[val_idx], y.iloc[val_idx]

    roc_auc = ROC_AUC((X_val, y_val))

    

    model.fit(X_dev, y_dev, validation_data=(X_val, y_val), epochs=40, batch_size=256, verbose=0, callbacks=[roc_auc])

    val_preds = model.predict(X_val, batch_size=5000)

    y_preds_nn[val_idx] = val_preds.flatten()

roc_auc_score(y, y_preds_nn)