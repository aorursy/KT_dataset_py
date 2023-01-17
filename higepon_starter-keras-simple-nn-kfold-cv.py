import numpy as np

import pandas as pd



from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.models import load_model

from keras.models import Sequential, Model

from keras.layers import Input, Dense, Dropout, Activation

from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

import os

print(os.listdir("../input"))
# Set this True when you want to check quickly if it works as expected.

# It will run with very small subset of whole data.

is_debug = False



# Load the data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



# We only use very small subset of data if is_debug.

if is_debug:

    train_df = train_df[0:300]

    test_df = test_df[0:300]    
# Remove unnecessary data. Well ID_code might have some leak, but we don't deep dive for now :)

X_train = train_df.drop(['target', 'ID_code'], axis=1)

X_test = test_df.drop(['ID_code'], axis=1)
# We scale both train and test data so that our NN works better.

sc = StandardScaler()

std = sc.fit_transform(X_test + X_train)
X_train_std = sc.fit_transform(X_train)

X_test_std = sc.fit_transform(X_test)
# This is the grand truth for training data.

Y = train_df[['target']]
# https://www.kaggle.com/tilii7/keras-averaging-runs-gini-early-stopping

# Our submission will be evaluated based on AUC.

class roc_auc_callback(Callback):

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

        y_pred = self.model.predict_proba(self.x, verbose=0)

        roc = roc_auc_score(self.y, y_pred)

        logs['roc_auc'] = roc_auc_score(self.y, y_pred)

        logs['norm_gini'] = ( roc_auc_score(self.y, y_pred) * 2 ) - 1



        y_pred_val = self.model.predict_proba(self.x_val, verbose=0)

        roc_val = roc_auc_score(self.y_val, y_pred_val)

        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)

        logs['norm_gini_val'] = ( roc_auc_score(self.y_val, y_pred_val) * 2 ) - 1



        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (str(round(roc,5)),str(round(roc_val,5)),str(round((roc*2-1),5)),str(round((roc_val*2-1),5))), end=10*' '+'\n')

        return



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return
# Very simple Neural Network model.

# This can be improved by many ways. e.g., more layers, batch normalization and etc.

def build_model():

    model = Sequential()

    model.add(Dense(units=64, input_dim=len(X_train.columns)))

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])

    return model
# Some parameters which control our training.

n_splits = 5

n_epochs = 10

patience = 10
# We do simple KFold Cross validation

y_test  = np.zeros((len(test_df)))

y_train = np.zeros((len(X_train_std)))



splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=10).split(X_train_std, Y))

for i, (train_idx, valid_idx) in enumerate(splits):    

    x_train_fold = X_train_std[train_idx]

    y_train_fold = Y.loc[train_idx]

    x_val_fold = X_train_std[valid_idx]

    y_val_fold = Y.loc[valid_idx]

    

    model = build_model()

    callbacks = [

        roc_auc_callback(training_data=(x_train_fold, y_train_fold),validation_data=(x_val_fold, y_val_fold)),

        EarlyStopping(monitor='norm_gini_val', patience=patience, mode='max', verbose=1),

    ]    

    model.fit(x_train_fold, y_train_fold, epochs=n_epochs, batch_size=256, callbacks=callbacks)



    y_val_preds = model.predict(x_val_fold)

    y_train[valid_idx] = y_val_preds.reshape(y_val_preds.shape[0])

    y_test_preds = model.predict(X_test_std)

    y_test += y_test_preds.reshape(y_test_preds.shape[0])



y_test = y_test / n_splits    
# This is our CV score.

roc_auc_score(Y, y_train)
submission = test_df[['ID_code']].copy()

submission['target'] = y_test

submission.to_csv('submission.csv', index=False)
!head submission.csv