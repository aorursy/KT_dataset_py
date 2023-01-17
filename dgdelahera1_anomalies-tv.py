import os

import random 

import numpy as np 

import pandas as pd 

import matplotlib.pylab as plt

from xgboost import XGBClassifier



from keras import objectives

from keras import backend as K

from keras.optimizers import Adam

from keras.models import Model, Sequential

from keras.losses import mse, binary_crossentropy

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.layers import Input, Dense, Lambda, BatchNormalization



from sklearn import preprocessing, metrics

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



print(os.listdir("../input"))
df_5 = pd.read_csv("../input/tv-data-5/channel_5.csv", sep=" ", header=None, names=("Time", "TV"))

df_5['Time'] = pd.to_datetime(df_5['Time'],unit='s')

df_5.head()

df_5.groupby(df_5.Time.dt.hour).mean().plot()
df_5['Time'] = pd.to_datetime(df_5['Time'])

df_5.index = df_5['Time'] 



days = df_5['TV'].resample('1D').sum().to_frame()

days = days.reset_index()

inactive = days[(days.TV <= 2000)]



print(inactive)
df_5['Time'] = pd.to_datetime(df_5['Time'])

df_5.index = df_5['Time'] 



grp = df_5['TV'].resample('1H').sum().to_frame()



print(grp.shape)

grp = grp.drop(grp[(grp.index.year == 2014) & (grp.index.month == 10) & (grp.index.day == 16)].index)

grp = grp.drop(grp[(grp.index.year == 2014) & (grp.index.month == 7) & (grp.index.day == 17)].index)

grp = grp.drop(grp[(grp.index.year == 2014) & (grp.index.month == 10) & (grp.index.day == 17)].index)

grp = grp.drop(grp[(grp.index.year == 2014) & (grp.index.month == 10) & (grp.index.day == 18)].index)

grp = grp.drop(grp[(grp.index.year == 2014) & (grp.index.month == 10) & (grp.index.day == 19)].index)

print(grp.shape)



grp.index = grp.index.map( lambda x: x.hour )

grp = grp.reset_index()

grp.head()
plt.figure(figsize=(12, 8))

plt.scatter(grp.Time, grp.TV)

plt.legend(loc='upper left')

plt.show()
anomalies = grp.copy()

anomalies["Time"] += 13

anomalies.TV = anomalies.TV.map(lambda x: x * random.uniform(0.25, 4.))

anomalies.Time = anomalies.Time.map(lambda x: (x - 24) if x >= 24 else x)



plt.figure(figsize=(12, 8))

plt.scatter(anomalies.Time, anomalies.TV)

plt.legend(loc='upper left')

plt.show()
plt.figure(figsize=(12, 8))

plt.scatter(grp.Time, grp.TV, color='g', linewidth='1', alpha=0.8, label='Normal')

plt.scatter(anomalies.Time, anomalies.TV, color='r', linewidth='1', alpha=0.8, label='Anomalies')

plt.legend(loc='upper left')

plt.show()
labels = np.zeros((grp.shape[0] + anomalies.shape[0]), dtype=int)



horas = np.append(grp.Time.values, anomalies.Time.values)

watios =  np.append(grp.TV.values, anomalies.TV.values)



x= np.column_stack((horas.astype(int),watios))



labels[grp.shape[0]:,] = 1



X = np.nan_to_num(x, copy=True)



x_scale = preprocessing.MinMaxScaler().fit_transform(X)



x_norm, x_fraud = x_scale[labels == 0], x_scale[labels == 1]



print("Done")
########################### VARATIONAL AUTOENCODER ############################



hidden_size = 16 #size of the hidden layer in encoder and decoder

latent_dim = 2 #number of latent variables to learn



input_dim = X.shape[1]



x = Input(shape=(input_dim, ))

t = BatchNormalization()(x)

t = Dense(hidden_size, activation='tanh' , name='encoder_hidden')(t)

t = BatchNormalization()(t)



z_mean = Dense(latent_dim, name='z_mean')(t)

z_log_var = Dense(latent_dim, name='z_log_var')(t)



def sampling(args):

    z_mean, z_log_var = args

    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.)

    return z_mean + K.exp(z_log_var / 2) * epsilon



z = Lambda(sampling, name='z_sampled')([z_mean, z_log_var])

#t = BatchNormalization()(z)



t = Dense(hidden_size, activation='tanh', name='decoder_hidden')(z)

#t = BatchNormalization()(t)



decoded_mean = Dense(input_dim, activation=None, name='decoded_mean')(t)



vae = Model(x, decoded_mean)



def rec_loss(y_true, y_pred):

    return K.sum(K.square(y_true - y_pred), axis=-1)



def kl_loss(y_true, y_pred):

    return - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)



def vae_loss(x, decoded_mean):

    rec_loss = K.sum(K.square(x - decoded_mean), axis=-1)

    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    return K.mean((rec_loss + kl_loss) / 2)



vae.compile(optimizer=Adam(lr=1e-2), loss=vae_loss, metrics=[rec_loss, kl_loss])

vae.summary()
n_epochs = 1000

batch_size = 128



early_stopping = EarlyStopping(monitor='val_kl_loss', patience=7, min_delta=1e-5) #stop training if loss does not decrease with at least 0.00001

reduce_lr = ReduceLROnPlateau(monitor='val_kl_loss', patience=4, min_delta=1e-5, factor=0.2) #reduce learning rate (divide it by 5 = multiply it by 0.2) if loss does not decrease with at least 0.00001



callbacks = [early_stopping, reduce_lr]



#collect training data in history object

history = vae.fit(x_norm, x_norm, 

                  batch_size=batch_size, epochs=n_epochs, 

                  callbacks=callbacks,

                  shuffle = True, validation_split = 0.20,

                  verbose = 2)



encoder = Model(x, z_mean)



norm_hid_rep = encoder.predict(x_norm)

fraud_hid_rep = encoder.predict(x_fraud)



plt.figure(figsize=(12, 8))

plt.scatter(norm_hid_rep[:,0],norm_hid_rep[:,1], color='g', linewidth='1', alpha=0.8, label='Normal')

plt.scatter(fraud_hid_rep[:,0],fraud_hid_rep[:,1], color='r', linewidth='1', alpha=0.8, label='Anomalies')

plt.legend(loc='upper left')

plt.show()


X = np.append(norm_hid_rep, fraud_hid_rep, axis=0)

y = np.append(np.zeros((norm_hid_rep.shape[0],), dtype=int), np.ones(fraud_hid_rep.shape[0],dtype=int))



p = np.random.permutation(len(X))

X = X[p]

y = y[p]



splitter = int(len(X)*0.9)



x_train = X[:splitter,]

y_train = y[:splitter,]



x_test = X[splitter:,]

y_test = y[splitter:,]



print("--------------------RF------------------------------")

clf = RandomForestClassifier()

clf.fit(x_train, y_train)

pred = clf.predict(x_test)



print("AUC(ROC): " + str(metrics.roc_auc_score(y_test, pred)))

print("Precision: " + str(metrics.precision_score(y_test, pred)))

print("Recall: " + str(metrics.recall_score(y_test, pred)))

print("F1 score: " + str(metrics.f1_score(y_test, pred)))



tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()



print("False positives: " + str(fp))

print("True positives: " + str(tp))

print("False negatives: " + str(fn))

print("True negatives: " + str(tn))



print("--------------------LR------------------------------")

clf = LinearDiscriminantAnalysis()

clf.fit(x_train, y_train)

pred = clf.predict(x_test)



print("AUC(ROC): " + str(metrics.roc_auc_score(y_test, pred)))

print("Precision: " + str(metrics.precision_score(y_test, pred)))

print("Recall: " + str(metrics.recall_score(y_test, pred)))

print("F1 score: " + str(metrics.f1_score(y_test, pred)))



tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()



print("False positives: " + str(fp))

print("True positives: " + str(tp))

print("False negatives: " + str(fn))

print("True negatives: " + str(tn))

print("--------------------XGB------------------------------")

clf = XGBClassifier(colsample_bytree= 0.7, learning_rate= 0.03, max_depth= 5, min_child_weight= 4, 

                   n_estimators= 500, nthread= 4, objective='reg:linear', 

                   silent= 1, subsample= 0.7)



clf.fit(x_train, y_train)

pred = clf.predict(x_test)



print("AUC(ROC): " + str(metrics.roc_auc_score(y_test, pred)))

print("Precision: " + str(metrics.precision_score(y_test, pred)))

print("Recall: " + str(metrics.recall_score(y_test, pred)))

print("F1 score: " + str(metrics.f1_score(y_test, pred)))



tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()



print("False positives: " + str(fp))

print("True positives: " + str(tp))

print("False negatives: " + str(fn))

print("True negatives: " + str(tn))
xgb1 = XGBClassifier()

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['reg:linear'],

              'learning_rate': [.03, 0.05, .07], #so called `eta` value

              'max_depth': [5, 6, 7],

              'min_child_weight': [4],

              'silent': [1],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [500]}



xgb_grid = GridSearchCV(xgb1,

                        parameters,

                        cv = 2,

                        n_jobs = 5,

                        verbose=True)



xgb_grid.fit(x_train, y_train)



print(xgb_grid.best_score_)

print(xgb_grid.best_params_)



params = xgb_grid.best_params_

model = XGBClassifier(**params)

model.fit(x_train, y_train)

pred = model.predict(x_test)



print("AUC(ROC): " + str(metrics.roc_auc_score(y_test, pred)))

print("Precision: " + str(metrics.precision_score(y_test, pred)))

print("Recall: " + str(metrics.recall_score(y_test, pred)))

print("F1 score: " + str(metrics.f1_score(y_test, pred)))



tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()



print("False positives: " + str(fp))

print("True positives: " + str(tp))

print("False negatives: " + str(fn))

print("True negatives: " + str(tn))