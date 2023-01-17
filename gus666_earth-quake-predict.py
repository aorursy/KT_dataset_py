# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import lightgbm as lgb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# import torch



# print(os.listdir())

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# try:

#     del train_full

# except:

#     pass

# train_full = pd.read_csv("../input/train.csv", nrows=100000000,dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

# train_full.rename({'acoustic_data': "signal", 'time_to_failure': "quaketime"}, inplace=True, axis="columns")
# try:

#     del data_loader

# except:

#     pass

# data_loader = preprocessing(nrows=10000000, ask_valid=False)

# data_loader.plot_train_data()


class myscaler:

    def __init__(self, data):

        self.shape = data.shape

        if len(data.shape) == 3:

            self.diff_stat = []

            self.scaler(data)

            self.transform = self._transform

        elif len(data.shape) == 2:

            scaler = StandardScaler()

            scaler.fit(data)

            self.transform = scaler.transform

        elif len(data.shape == 1):

            print("dim error")

            self.transform = None

        

    def scaler(self, data):

        features = data.shape[-1]

        for i in range(features):

            self.diff_stat.append([np.mean(data[...,i]), np.std(data[...,i])])

        

    def _transform(self, data):

        assert data.shape[-1] == self.shape[-1]

        features = data.shape[-1]

        for i in range(features):

             data[...,i] = (data[...,i] - self.diff_stat[i][0])/self.diff_stat[i][1]

        return data

                                   

    def transform(self,data):

        return self.transform(data)

        

def gen_feature(data):

        features = []

        features.append(data.abs().min())

        features.append(data.abs().max())

        features.append(data.abs().mean())

        features.append(data.min())

        features.append(data.max())

        features.append(data.mean())

        features.append(data.kurtosis())

        features.append(data.skew())

        for i in np.arange(0.01, 0.1, 0.01):

            features.append(np.quantile(data, i))

        for i in np.arange(0.9, 1, 0.01):

            features.append(np.quantile(data, i))

        data = data.diff().dropna()

        features.append(data.abs().min())

        features.append(data.abs().max())

        features.append(data.abs().mean())

        features.append(data.min())

        features.append(data.max())

        features.append(data.mean())

        features.append(data.kurtosis())

        features.append(data.skew())

        for i in np.arange(0.01, 0.1, 0.01):

            features.append(np.quantile(data, i))

        for i in np.arange(0.9, 1, 0.01):

            features.append(np.quantile(data, i))

        return pd.Series(features)



def rolling_feature(data, window_size = 2000, step = 2000):

    def feature_small_set(data):

        features = []

        features.append(data.min())

        features.append(data.max())

        features.append(data.mean())

        features.append(data.std())

        data = np.diff(data)

        features.append(data.min())

        features.append(data.max())

        features.append(data.mean())

        features.append(data.std())

#         features.append(data.kurtosis())

#         features.append(data.skew())

#         for i in np.arange(0.01, 0.1, 0.01):

#             features.append(np.quantile(data, i))

        for i in np.arange(0.9, 1, 0.01):

            features.append(np.quantile(data, i))

        return features

    ans = []

    for i in range(window_size, len(data), step):

        ans.append(feature_small_set(data[(i-window_size):i]))

    return np.array(ans)

    

def check_quake(data):

    return (data.diff(-1).dropna() >= 0).all()





def load_batch(data,labels, batch_size=20, shuffle=True):

    assert len(data) == len(labels)

    size = len(data)

    for i in range(0, size, batch_size):

        if(i+batch_size > size):

            yield data[i:size], labels[i:size]

        else:

            yield data[i:i+batch_size], labels[i:i + batch_size]
features_name = ["abs_min", "abs_max", "abs_mean", "min", 'max', 'mean', 'kurtosis', 'skew'] + ["quantile" + f"{i:.3}" for i in np.arange(0.01, 0.1, 0.01)] + ["quantile" + f"{i:.3}" for i in np.arange(0.9, 1, 0.01)]

    

features_name = features_name + ["diff_" + i for i in features_name ]
def gen_data(feature_fun = gen_feature, test_size=100, size = None, shuffle=True, start_from = None):

    train_gen = pd.read_csv("../input/train.csv", chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

    X_train = []

    y_train = []

    X_test = []

    y_test = []

    import gc

    if start_from is None:

        start_from = 0

    for i, data in enumerate(train_gen):

        data.rename({'acoustic_data': "signal", 'time_to_failure': "quaketime"}, inplace=True, axis="columns")

        if i < start_from:

            del data

            continue

        if size is not None and i >= start_from + size:

            break

        if check_quake(data["quaketime"]) == False:

            print("quake at", i, "excluded")

            try:

                print(X_train[-1].shape)

            except IndexError:

                pass

        X_train.append(feature_fun(data.signal))

        y_train.append(data.quaketime.values[-1])

        del data

        gc.collect()

    X_train = np.array(X_train)

    y_train = np.array(y_train)

    if shuffle:

        index = np.arange(0, len(X_train))

        np.random.shuffle(index)

        X_train = X_train[index]

        y_train = y_train[index]

    X_test = X_train[:test_size]

    y_test = y_train[:test_size]

    X_train = X_train[test_size:]

    y_train = y_train[test_size:]

    print(X_train.shape, y_train.shape)

    print(X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = gen_data(size=2000,test_size=200)
plt.plot(y_train)

plt.plot(y_test)
from catboost import CatBoostRegressor, Pool

#data scaling

from sklearn.preprocessing import StandardScaler

#hyperparameter optimization

from sklearn.model_selection import GridSearchCV

#support vector machine model

from sklearn.svm import NuSVR, SVR

#kernel ridge model

from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import RandomForestRegressor



scaler1 = StandardScaler()

scaler1.fit(X_train)

X_train_scaled = scaler1.transform(X_train)

X_test = scaler1.transform(X_test)
train_pool = Pool(X_train_scaled, y_train)

cat_models = CatBoostRegressor(iterations=10000, loss_function="MAE", boosting_type="Ordered")

cat_models.fit(X_train_scaled, y_train, silent=True)

print(cat_models.best_score_)



test_pool = Pool(X_test, y_test)

pre = cat_models.predict(X_test)

print("catboost score:", np.mean(np.abs(y_test - pre)))
rf = RandomForestRegressor(max_depth=2, n_estimators=3000,max_features=40, criterion="mae")

rf.fit(X_train_scaled, y_train)

import os

from collections import OrderedDict

from pprint import pprint

important = rf.feature_importances_

print(len(important), len(features_name))

pprint(OrderedDict(zip(features_name, important)))
print("rf score:", rf.score(X_test, y_test.flatten()))


# parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],

#                'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]

#                #'nu': [0.75, 0.8, 0.85, 0.9, 0.95, 0.97]}]



# reg1 = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_absolute_error')

# reg1.fit(X_train_scaled, y_train.flatten())

# y_pred1 = reg1.predict(X_train_scaled)

# # {'C': 2, 'gamma': 0.001}

# print("Best CV score: {:.4f}".format(reg1.best_score_))

# print(reg1.best_params_)





svm = SVR(kernel="rbf", tol=0.01, C=2, gamma=0.001)

svm.fit(X_train_scaled, y_train)

print("svm score:", np.mean(np.abs(y_test - svm.predict(X_test))))
# para = [{#'degree':[3,5,7,9], 

#          "nu":[0.3, 0.5, 0.7, 0.9],

#         "C":[0.2, 0.5, 1, 2]}]







# svm = GridSearchCV(NuSVR(kernel="poly", degree=3,gamma="scale"),para, cv=5, scoring='neg_mean_absolute_error')

# svm.fit(X_train_scaled, y_train.flatten())

# y_pred = svm.predict(X_train_scaled)
params = {'num_leaves': 128,

          'min_child_samples': 79,

          'objective': 'gamma',

          'max_depth': -1,

          'learning_rate': 0.01,

          "boosting_type": "gbdt",

          "subsample_freq": 5,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.1302650970728192,

          'reg_lambda': 0.3603427518866501,

          'colsample_bytree': 0.1

         }

lgbm = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1)

lgbm.fit(X_train_scaled, y_train, eval_set=[(X_train_scaled, y_train), (X_test, y_test)], eval_metric='mae',

                    verbose=10000)

print("lgbm score:", np.mean(np.abs(y_test - lgbm.predict(X_test))))
# y_pred1 = reg1.predict(X_train_scaled)

# y_pred2 = cat_models.predict(X_train_scaled)

# y_pred3 = rf.predict(X_train_scaled)

# y_pred4 = lgbm.predict(X_train_scaled)

# y = (y_pred1 + y_pred2 + y_pred3)/3

# fig, ax = plt.subplots(5, 1, figsize=(14,6))

# plt.figure(figsize=(8, 6))

# ax[0].scatter(y_train.flatten(), y_pred1, c='b')

# ax[0].plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

# ax[1].scatter(y_train.flatten(), y_pred2, c='r')

# ax[1].plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

# ax[2].scatter(y_train.flatten(), y_pred3,c="g")

# ax[2].plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

# ax[3].scatter(y_train.flatten(), y, c='g')

# ax[3].plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

# plt.show()
def predict(models, x, weights = None):

    ans = []

    y = 0

    for i, model in enumerate(models):

        if weights is None:

            y += model.predict(x)*(1/len(models))

        else:

            y += model.predict(x)*weights[i]

    return y

models = [cat_models, svm, rf, lgbm]
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

index = list(submission.index)
test = []

for seg_id in index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    test.append(gen_feature(seg['acoustic_data']))

test = np.array(test)

test = scaler1.transform(test)
y_pre1 = predict(models, test, [0.3,0.3,0.2,0.2])
try:

    del X_train, y_train, X_test, y_test

except:

    pass

X_train, y_train, X_test, y_test = gen_data(feature_fun = rolling_feature,size=2000,test_size=200)


from keras.models import Sequential

from keras.layers import Dense, Conv1D, LSTM, Dropout, BatchNormalization, LeakyReLU, CuDNNGRU

from keras.optimizers import Adam

from keras.models import Model

from keras import regularizers

import keras

from keras.utils import plot_model

from sklearn.metrics import mean_absolute_error

from keras.callbacks import EarlyStopping



class MyLSTM:



    def __init__(self, data, labels, valid_data, valid_labels,input_shape = 149):

        # return super().__init__(*args, **kwargs)

        self.nn = self.bulid_lstm(data.shape[1], data.shape[2])

        cb = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='min')

        self.nn.compile(optimizer="adam", loss="MAE")

        self.input_shape = (data.shape[1], data.shape[2])

        self.data = data

        self.labels = labels

        self.valid = (valid_data,valid_labels)



    def bulid_lstm(self, input_shape, feature_shape):

        seq = keras.Input(shape=(input_shape, feature_shape))

#         x = Conv1D(32, kernel_size=7, strides=2, kernel_regularizer=regularizers.l1(0.01))(seq)

#         x = Conv1D(8, kernel_size=7, strides=2, padding="valid", kernel_regularizer=regularizers.l1(0.01))(x)

#         x = BatchNormalization()(x)

#         x = LeakyReLU()(x)

#         x = Dropout(rate=0.2)(x)

#         x = LSTM(units=32, return_sequences=True, activation='tanh')(x)

#         x = Dropout(rate=0.2)(x)

#         x = LSTM(units=16, return_sequences=True)(x)

#         x = Dropout(0.2)(x)

#         x = LSTM(units=50, return_sequences=True)(x)

#         x = Dropout(0.2)(x)

        x = CuDNNGRU(48)(seq)

#         x = LSTM(units=8)(x)

#         x = Dropout(rate=0.2)(x)

        x = Dense(10, activation='relu')(x)

        x = Dense(units=1)(x)

        model = Model(inputs=seq, outputs=x)

        model.summary()

        # plot_model(model, to_file="see.jpg")

        # model.compile(optimizer="rmsprop", loss="MAE")

        return model



    def train_on_batch(self, epochs = 50, batch_size=20, intervals=10):

        losses = []

        test_losses = []

        for epoch in range(epochs):

            for batch_i, data in enumerate(load_batch(self.data, self.labels, batch_size=batch_size)):

#                 print(data[0].shape)

                loss = self.nn.train_on_batch(data[0], data[1],)# validation_data=(self.valid[0], self.valid[1]))

                losses.append(loss)

                valid = np.mean(np.abs(self.valid[1] - self.nn.predict(self.valid[0])))

                test_losses.append(np.mean(valid))

                if batch_i%intervals == 0 or (batch_i == batch_size-1):

                    print(f"[Epoch {epoch}/{epochs}] [Batch {batch_i}] [MAE loss {loss}] [valid loss {valid}]")

            plt.plot(losses, label="training loss")

            plt.plot(test_losses, label="validation loss")

            plt.legend()

            plt.show()

            if epoch > 4:

                stop = (test_losses[-1] > np.min(test_losses) + 0.1)

                if stop:

                    print(f"stop train at [Epoch {epoch}/{epochs}]")

                    break



    def predict(self, test_data):

        return self.nn.predict(test_data)

lstm = MyLSTM(X_train, y_train, X_test, y_test)
lstm.train_on_batch(batch_size=50,epochs=50)
plt.plot(y_test)

plt.scatter(list(range(len(y_test))),lstm.predict(X_test))

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

index = list(submission.index)
test = []

for seg_id in index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    test.append(rolling_feature(seg['acoustic_data']))

test = np.array(test)

test = scaler.transform(test)

y_pre2 = lstm.predict(test)
y_pre = 0.6*y_pre1 + 0.4*y_pre
submission.time_to_failure = y_pre

submission.to_csv('submission.csv')