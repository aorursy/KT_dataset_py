# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, StratifiedKFold 

from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#from sklearn.linear_model import *

#from sklearn.decomposition import PCA

import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
from keras.models import Model

from keras.layers import *

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam
df = pd.read_csv('../input/train.csv')

df.head()
X = df.values[:,2:]

mmscaler = MinMaxScaler()

X = mmscaler.fit_transform(X)

#pca = PCA(100, svd_solver='full')

#pca.fit(X)

y = df.values[:,1]

print(X.shape, y.shape)
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
def simple_model(input_shape):

    inp = Input(shape=(input_shape[1],))

    x = Dense(150, activation='relu')(inp)

    x = Dropout(0.3)(x)

    x = Dense(30, activation='relu')(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model
N_SPLITS = 10



splits = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True).split(X, y))

preds_val = []

y_val = []

best_models = []



for idx, (train_idx, val_idx) in enumerate(splits):

    print("Beginning fold {}".format(idx+1))

    X_train, y_train, X_val, y_val = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    model = simple_model(X_train.shape)

    cb = ModelCheckpoint('weights.h5', monitor='val_acc', mode='max', save_best_only=True, save_weights_only=True)

    model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[cb], verbose=0)

    model.load_weights('weights.h5')

    score = roc_auc_score(y_val, model.predict(X_val))

    print((model, score))

    best_models.append((model, score))

    
df_test = pd.read_csv('../input/test.csv')

print(len(df_test))

df_test.head()
X_test = df_test.values[:,1:]

X_test = mmscaler.transform(X_test)
y_preds = []

for mod, score in best_models:

    y_preds.append(mod.predict(X_test))

y_preds = np.concatenate(y_preds, axis=1)

y_preds.shape
subs = pd.read_csv('../input/sample_submission.csv')

y_preds = y_preds.mean(axis=1)

subs['target'] = y_preds

subs.to_csv('submission.csv', index=False)