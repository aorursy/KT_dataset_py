# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

# import iplot
sample_submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

train_features.head()
train_features.shape, train_targets_scored.shape, test_features.shape
train_features.info()
train_targets_scored.info()
train_features.sig_id.nunique(), train_targets_scored.sig_id.nunique()
train_features.cp_type.value_counts(normalize=True).plot(kind='pie', figsize=(12, 5), fontsize=12,

                                                         title='CP Type', autopct='%1.1f%%')

plt.show()
train_features.cp_time.value_counts(normalize=True).plot(kind='bar', figsize=(12, 5), fontsize=14,

                                                         title='CP Time', xlabel='Time')

plt.show()
train_features.cp_dose.value_counts(normalize=True).plot(kind='bar', figsize=(12, 5), fontsize=14, 

                                                         title='CP Dose', xlabel='Dose')

plt.show()
gcols = [col for col in train_features.columns if 'g-' in col]

ccols = [col for col in train_features.columns if 'c-' in col]
g = sns.pairplot(train_features[gcols[:10]])

plt.show()
g = sns.pairplot(train_features[ccols[:10]])

plt.show()
train_target_count = train_targets_scored.sum()[1:].sort_values()
train_target_count[:50].plot(kind='barh', title='Least Target Occurances', fontsize='14', figsize=(5, 20))

plt.show()
train_target_count[-50:].plot(kind='barh', title='Most Target Occurences', fontsize='12', figsize=(12, 20))

plt.show()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 14))

for row in range(2):

    for col in range(2):

        random_int = np.random.randint(1, 700)

        train_features.loc[:2000, 'g-'+str(random_int)].plot(ax=ax[row][col], title='G - '+str(random_int))

plt.show()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 14))

for row in range(2):

    for col in range(2):

        random_int = np.random.randint(1, 100)

        train_features.loc[:2000, 'c-'+str(random_int)].plot(ax=ax[row][col], title='C - '+str(random_int), label='Train')

        # test_features.loc[:2000, 'c-'+str(random_int)].plot(ax=ax[row][col], title='C - '+str(random_int), label='Test')

plt.show()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 14))

for row in range(2):

    for col in range(2):

        random_int = np.random.randint(1, 700)

        f = sns.boxplot(train_features.loc[:2000, 'g-'+str(random_int)], ax=ax[row][col])

        f.set_title('G - '+str(random_int))

plt.show()
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 14))

for row in range(2):

    for col in range(2):

        random_int = np.random.randint(1, 100)

        f = sns.boxplot(train_features.loc[:2000, 'c-'+str(random_int)], ax=ax[row][col])

        f.set_title('C - '+str(random_int))

plt.show()
plt.figure(figsize=(100, 100))

sns.heatmap(train_features[ccols].corr())

plt.show()
# gcols
# Check correlation

cols = gcols+ccols

correlation = train_features[cols].corr()
len(cols), correlation.shape
gc_corr = {}



for i, c1 in enumerate(cols):

    for j, c2 in enumerate(cols):

        if i < j:

            corr = correlation.iloc[i, j]

            if corr >= 0.8:

                gc_corr[c1] = c2
useful_col = cols - gc_corr.keys()

useful_col = list(useful_col)

useful_col = useful_col + ['cp_type', 'cp_dose', 'cp_time']
# useful_col
len(useful_col), len(cols)
ctl_vehicle_id = train_features[train_features.cp_type == 'ctl_vehicle']['sig_id']



ctl_vehicle_id = list(ctl_vehicle_id)



sum_target_cp_type_ctl_vehicle = train_targets_scored[train_targets_scored.sig_id.isin( ctl_vehicle_id)].sum()[1:].sum()



print('Training Data - For cp_type - ctl_vehicle total sum of targets : {}'.format(sum_target_cp_type_ctl_vehicle))
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# labelencoder = LabelEncoder()



# train_features['cp_type_encoded'] = labelencoder.fit_transform(train_features.cp_type)

# test_features['cp_type_encoded'] = labelencoder.transform(test_features.cp_type)
train_features.columns
# train_features['cp_time'] = train_features.cp_time.map({24:0, 48:1, 72:2})

# test_features['cp_time'] = test_features.cp_time.map({24:0, 48:1, 72:2})
train_features['cp_type'] = train_features.cp_type.map({'trt_cp':0, 'ctl_vehicle':1})

test_features['cp_type'] = test_features.cp_type.map({'trt_cp':0, 'ctl_vehicle':1})
train_features['cp_dose'] = train_features.cp_dose.map({'D1':0, 'D2':1})

test_features['cp_dose'] = test_features.cp_dose.map({'D1':0, 'D2':1})
train_features.cp_type.value_counts()
# # train_cp_time = pd.get_dummies(train_features.cp_time, drop_first=True)



# train_features['cp_dose_encoded'] = labelencoder.fit_transform(train_features.cp_dose)



# # train_features = pd.concat([train_features, train_cp_time], axis = 1)



# # test_cp_time = pd.get_dummies(test_features.cp_time, drop_first=True)

# test_features['cp_dose_encoded'] = labelencoder.fit_transform(test_features.cp_dose)



# # test_features = pd.concat([test_features, test_cp_time], axis = 1)
train_features.shape, test_features.shape
import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')



from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Input, BatchNormalization

from keras.optimizers import RMSprop, Adam, SGD, Adamax

from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from keras.layers.advanced_activations import PReLU

from keras.regularizers import l1, l2, l1_l2

from kerastuner.tuners import RandomSearch



from tensorflow_addons.layers import WeightNormalization



import matplotlib.pyplot as plt



from sklearn.metrics import log_loss

from sklearn.model_selection import KFold
# class ClassificationReport(Callback):

    

#     def __init__(self, train_data=(), validation_data=()):

        

#         super(Callback ,self).__init__()

        

#         self.X_train, self.Y_train = train_data

#         self.X_val, self.Y_val = validation_data

        

#         self.train_log_loss = []

#         self.val_log_loss = []

        

#     def on_epoch_end(self, epoch, log={}):

        

#         train_prediction = np.round(self.model.predict(self.X_train, verbose=0))

#         val_prediction = np.round(self.model.predict(self.X_val, verbose=0))

        

#         # training log loss

#         train_loss = []

#         for i, col in enumerate(self.Y_train.columns):

#             # print(self.Y_train.loc[:, col].values.shape, train_prediction[:, i].shape)

#             loss = log_loss(self.Y_train.loc[:, col].values, train_prediction[:, i].astype(float), labels=[0, 1])

#             train_loss.append(loss)

#         self.train_log_loss.append(np.mean(train_loss))

        

#         # validation log loss

#         val_loss = []

#         for i, col in enumerate(self.Y_val.columns):

#             loss = log_loss(self.Y_val.loc[:, col].values, val_prediction[:, i].astype(float), labels=[0, 1])

#             val_loss.append(loss)

#         self.val_log_loss.append(np.mean(val_loss))

        

#         print("\n Epoch - {}, Training Log Loss - {:.6}, Validation Log Loss - {:.6} \n".format(epoch+1,

#                                                                                                 np.mean(train_loss),

#                                                                                                 np.mean(val_loss)))
# class MoA:

    

#     def __init__(self, X, Y, hp, folds=2, learning_rate=0.0001, dropout=0.1, seed=141, batch_size=128, epochs=10):

        

#         self.X = X

#         self.Y = Y

#         self.folds = folds

#         self.learning_rate = learning_rate

#         self.dropout = dropout

#         self.seed = seed

#         self.batch_size = batch_size

#         self.epochs = epochs

#         self.models = []

#         self.scores = {}

#         self.hp = hp

    

#     def build_model(self):

        

#         inp = Input(shape=(self.X.shape[1], ))

#         batch_norm = BatchNormalization()(inp)

        

#         x1 = WeightNormalization(Dense(units=self.hp.Int('units',

#                                                     min_value = 1024,

#                                                     max_value = 4096,

#                                                     step = 128),

#                                        activation='selu'))(batch_norm)

#         drop = Dropout(self.hp.Choice('learning_rate', values=[0.2, 0.3, 0.4, 0.5, 0.6]))(x1)

#         batch_norm = BatchNormalization()(drop)

        

        

#         x1 = WeightNormalization(Dense(units=self.hp.Int('units',

#                                                     min_value = 512,

#                                                     max_value = 2048,

#                                                     step = 128),

#                                        activation='selu'))(batch_norm)

#         drop = Dropout(self.hp.Choice('learning_rate', values=[0.2, 0.3, 0.4, 0.5, 0.6]))(x1)

#         batch_norm = BatchNormalization()(drop)

        

#         x1 = WeightNormalization(Dense(units=self.hp.Int('units',

#                                                     min_value = 256,

#                                                     max_value = 1024,

#                                                     step = 128),

#                                        activation='selu'))(batch_norm)

#         drop = Dropout(self.hp.Choice('learning_rate', values=[0.2, 0.3, 0.4, 0.5, 0.6]))(x1)

#         batch_norm = BatchNormalization()(drop)

        

#         x1 = WeightNormalization(Dense(units=self.hp.Int('units',

#                                                     min_value = 256,

#                                                     max_value = 512,

#                                                     step = 128),

#                                        activation='selu'))(batch_norm)

#         drop = Dropout(self.hp.Choice('learning_rate', values=[0.2, 0.3, 0.4, 0.5, 0.6]))(x1)

#         batch_norm = BatchNormalization()(drop)

        

#         dense = Dense(self.Y.shape[1], activation='sigmoid')(batch_norm)

        

#         model = Model(inputs=inp, outputs=dense)

        

#         model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        

#         return model

    

    

#     def train_model(self):

        

#         mkf = MultilabelStratifiedKFold(n_splits=self.folds, random_state=seed, shuffle=True)

        

#         for fold, (train_idx, val_idx) in enumerate(mkf.split(self.X, self.Y)):

            

#             print('\n Fold - {}'.format(fold))

            

#             X_train = self.X.loc[train_idx, :]

#             Y_train = self.Y.loc[train_idx, :]

            

#             X_val = self.X.loc[val_idx, :]

#             Y_val = self.Y.loc[val_idx, :]

            

#             # print(Y_val[: 2])

            

#             metrics = ClassificationReport(train_data=(X_train, Y_train), validation_data=(X_val, Y_val))

            

#             model = self.build_model()

            

#             reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.15, patience=3, verbose=1,

#                                                epsilon=self.learning_rate, mode='min')

#             early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode= 'min')

            

#             history = model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs,

#                       callbacks=[reduce_lr_loss, early_stop], validation_data=(X_val, Y_val), verbose=1)

            

#             self.models.append(model)

            

#             self.scores[fold] = {

#                 'training_log_loss' : metrics.train_log_loss,

#                 'validation_log_loss' : metrics.val_log_loss

#             }

            

    

#     def plot_learning_curve(self):

        

#         fig, ax = plt.subplots(nrows=self.folds, ncols=2, figsize=(20, self.folds * 6), dpi=100)

        

#         for i in range(self.folds):

            

#             sns.lineplot(x=np.arange(1, self.epochs+1), y=self.models[i].history.history['loss'], ax=ax[i][0],

#                         label='Train Loss')

#             sns.lineplot(x=np.arange(1, self.epochs+1), y=self.models[i].history.history['val_loss'], ax=ax[i][0],

#                         label='Validation Loss')

            

#             sns.lineplot(x=np.arange(1, self.epochs+1), y=self.scores[i]['training_log_loss'], ax=ax[i][1],

#                         label='Train Log-Loss')

#             sns.lineplot(x=np.arange(1, self.epochs+1), y=self.scores[i]['validation_log_loss'], ax=ax[i][1],

#                         label='Validation Log-Loss')

            

#             for j in range(self.folds):

#                 ax[i][j].legend()

#                 ax[i][j].set_xlabel('Epoch', size=12)

#                 ax[i][j].tick_params(axis='x', labelsize=12)

#                 ax[i][j].tick_params(axis='y', labelsize=12)

            

#     def predict(self, X_predict):

        

#         Y_predict = np.zeros((X_predict.shape[0], self.Y.shape[1]))

        

#         for i in range(self.folds):

            

#             temp_predict = self.models[i].predict(X_predict)

            

#             Y_predict = (Y_predict + temp_predict)/self.folds

        

#         return Y_predict
# def build_model(hp):

        

#         inp = Input(shape=(X.shape[1], ))

#         batch_norm = BatchNormalization()(inp)

        

#         x1 = WeightNormalization(Dense(units=hp.Int('units_1',

#                                                     min_value = 1024,

#                                                     max_value = 4096,

#                                                     step = 128),

#                                        activation='elu'))(batch_norm)

#         drop = Dropout(hp.Choice('dropout', values=[0.2, 0.3, 0.4, 0.5, 0.6]))(x1)

#         batch_norm = BatchNormalization()(drop)

        

        

#         x1 = WeightNormalization(Dense(units=hp.Int('units_2',

#                                                     min_value = 512,

#                                                     max_value = 2048,

#                                                     step = 128),

#                                        activation='elu'))(batch_norm)

#         drop = Dropout(hp.Choice('dropout', values=[0.2, 0.3, 0.4, 0.5, 0.6]))(x1)

#         batch_norm = BatchNormalization()(drop)

        

#         x1 = WeightNormalization(Dense(units=hp.Int('units_3',

#                                                     min_value = 256,

#                                                     max_value = 1024,

#                                                     step = 128),

#                                        activation='elu'))(batch_norm)

#         drop = Dropout(hp.Choice('dropout', values=[0.2, 0.3, 0.4, 0.5, 0.6]))(x1)

#         batch_norm = BatchNormalization()(drop)        



#         x1 = WeightNormalization(Dense(units=hp.Int('units_4',

#                                                     min_value = 256,

#                                                     max_value = 512,

#                                                     step = 128),

#                                        activation='elu'))(batch_norm)

#         drop = Dropout(hp.Choice('dropout', values=[0.2, 0.3, 0.4, 0.5, 0.6]))(x1)

#         batch_norm = BatchNormalization()(drop)

        

#         dense = Dense(Y.shape[1], activation='sigmoid')(batch_norm)

        

#         model = Model(inputs=inp, outputs=dense)

        

#         model.compile(optimizer=Adam(), loss='binary_crossentropy')

        

#         return model
def build_model():

        

        inp = Input(shape=(X.shape[1], ))

        batch_norm = BatchNormalization()(inp)

        

        x1 = WeightNormalization(Dense(units=1024,

                                       activation='elu'))(batch_norm)

        drop = Dropout(0.5)(x1)

        batch_norm = BatchNormalization()(drop)

        

        

        x1 = WeightNormalization(Dense(640,

                                       activation='elu'))(batch_norm)

        drop = Dropout(0.5)(x1)

        batch_norm = BatchNormalization()(drop)

        

        x1 = WeightNormalization(Dense(768,

                                       activation='elu'))(batch_norm)

        drop = Dropout(0.5)(x1)

        batch_norm = BatchNormalization()(drop)        



        x1 = WeightNormalization(Dense(units=256,

                                       activation='elu'))(batch_norm)

        drop = Dropout(0.5)(x1)

        batch_norm = BatchNormalization()(drop)

        

        dense = Dense(Y.shape[1], activation='sigmoid')(batch_norm)

        

        model = Model(inputs=inp, outputs=dense)

        

        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        

        return model
def get_log_loss(Y, val_Y):    

    full_loss = []

    for i, col in enumerate(Y.columns):

        loss = log_loss(Y.loc[:, col].values, val_Y[:, i].astype(float), labels=[0, 1])

        full_loss.append(loss)



    loss = np.mean(full_loss)

    

    # print("\n Log Loss - {:.6} \n".format(loss))

    

    return loss






def train_model(X, Y, X_Test, folds = 3, seed=3, batch_size = 128, epochs = 50, learning_rate=1e-4):

    

    submission = sample_submission.drop('sig_id', axis=1).copy()

    submission.loc[:, :] = 0

    

    mkf = MultilabelStratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)



    final_log_loss = []

    

    T_logloss = []

    V_logloss = []

    

    for n in range(seed):

        

        for fold, (train_idx, val_idx) in enumerate(mkf.split(X, Y)):



            print('\n Run - {}, Fold - {}'.format(n, fold))



            X_train = X.loc[train_idx, :]

            Y_train = Y.loc[train_idx, :]



            X_val = X.loc[val_idx, :]

            Y_val = Y.loc[val_idx, :]



            model = build_model()



            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.15, patience=3, verbose=1,

                                               epsilon=learning_rate, mode='min')

            

            early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode= 'min')

            

#             checkpoint = ModelCheckpoint(monitor = 'val_loss', verbose = 0, 

#                               save_best_only = True, save_weights_only = True, mode = 'min')



            history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,

                      callbacks=[reduce_lr_loss, early_stop], validation_data=(X_val, Y_val), verbose=2)



            Train_Pred = model.predict(X_train)

            Val_pred = model.predict(X_val)

        

            Train_Pred = Train_Pred/(seed*(fold+1))

            Val_pred = Val_pred/(seed*(fold+1))

            

            train_logloss = get_log_loss(Y_train, Train_Pred)

            val_logloss = get_log_loss(Y_val, Val_pred)

        

            print('Training Log Loss : {:.6}'.format(train_logloss))

            print('Validation Log Loss : {:.6}'.format(val_logloss))



            T_logloss.append(train_logloss)

            V_logloss.append(val_logloss)

            

            submission += model.predict(X_Test)

            

            submission = submission/((fold+1)*seed)

    

        final_T_logloss = np.mean(T_logloss)

        final_V_logloss = np.mean(V_logloss)



        print('Final Training Log Loss : {:.6}'.format(final_T_logloss))

        print('Final Validation Log Loss : {:.6}'.format(final_V_logloss))

    return submission
X = train_features.drop('sig_id', axis = 1)

Y = train_targets_scored.drop('sig_id', axis = 1)



X_test = test_features.drop('sig_id', axis = 1)
submission = train_model(X, Y, X_test, folds = 3, seed=3, batch_size = 128, epochs = 50, learning_rate=1e-4)
sample_submission.iloc[:, 1:] = submission
sample_submission
sample_submission.to_csv('submission.csv', index=False)
# tuner.get_best_hyperparameters
# Y_predicted = np.where(Y_predicted < 0.5, 0, 1)
# submission = pd.concat([submission, submission1], axis=0)