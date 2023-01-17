# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split





from keras import layers, Input, Model, models

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import roc_auc_score

from keras import regularizers

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape
train.info()
train.head(10)
train.isnull().sum(axis=0).sum()
test.isnull().sum(axis=0).sum()
train.describe()
feature = train.drop("target", axis=1).drop("id", axis=1)
#plt.violinplot(train.drop("target", axis=1))

#plt.violinplot(train["muggy-smalt-axolotl-pembus"],train["jumpy-thistle-discus-sorted"])



f, ax = plt.subplots(5, 1, figsize=(20, 8), sharex=True)

#sns.violinplot(data=train.drop("target", axis=1),ax=ax)





for cname in feature.columns[0:50]:

    sns.distplot(feature[cname].sample(1000), ax=ax[0])

for cname in feature.columns[50:100]:

    sns.distplot(feature[cname].sample(1000), ax=ax[1])

for cname in feature.columns[100:150]:

    sns.distplot(feature[cname].sample(100), ax=ax[2])

for cname in feature.columns[150:200]:

    sns.distplot(feature[cname].sample(1000), ax=ax[3])

for cname in feature.columns[200:258]:

    sns.distplot(feature[cname].sample(1000), ax=ax[4])
feature.describe().sort_values(by='max',ascending=False,axis=1)     
f, ax = plt.subplots(1, 1, figsize=(20, 8), sharex=True)

sns.distplot(feature["wheezy-copper-turtle-magic"].sample(1000), ax=ax)
feature["wheezy-copper-turtle-magic"].sample(10)
f, ax = plt.subplots(5, 1, figsize=(20, 8), sharex=True)

#sns.violinplot(data=train.drop("target", axis=1),ax=ax)



feature_nolarge = feature.drop("wheezy-copper-turtle-magic", axis=1)



for cname in feature_nolarge.columns[0:50]:

    sns.distplot(feature_nolarge[cname].sample(1000), ax=ax[0])

for cname in feature_nolarge.columns[50:100]:

    sns.distplot(feature_nolarge[cname].sample(1000), ax=ax[1])

for cname in feature_nolarge.columns[100:150]:

    sns.distplot(feature_nolarge[cname].sample(1000), ax=ax[2])

for cname in feature_nolarge.columns[150:200]:

    sns.distplot(feature_nolarge[cname].sample(1000), ax=ax[3])

for cname in feature_nolarge.columns[200:257]:

    sns.distplot(feature_nolarge[cname].sample(1000), ax=ax[4])
trn_x, valid_x, trn_y, valid_y = train_test_split(train.drop(['id', 'target'], axis=1), train.target, random_state=33, test_size=0.15)

trn_x.shape, valid_x.shape, trn_y.shape, valid_y.shape

def build_model():

    inp = Input(shape=(trn_x.shape[1],), name='input')

    x = layers.Dense(1000, activation='relu')(inp)

    x = layers.Dense(750, activation='relu')(x)

    x = layers.Dense(500, activation='relu')(x)

    x = layers.Dense(1, activation='sigmoid')(x)

    

    model = Model(inp, x)

    model.compile(optimizer='adam',

                 loss='binary_crossentropy', metrics=['acc'])

    

    return model



model = build_model()

model.summary()


model = build_model()



weights_path = f'weights.best.hdf5'

val_loss_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, mode='min', min_lr=1e-6)



model.fit(trn_x, trn_y, epochs=80, validation_data=(valid_x, valid_y),

         callbacks=[val_loss_checkpoint, reduceLR], batch_size=512, verbose=1)
model.load_weights(weights_path)

val_preds = model.predict(valid_x, batch_size=2048, verbose=1)
roc_auc_score(valid_y.values, val_preds.reshape(-1))

trn_wheezy = pd.get_dummies(trn_x['wheezy-copper-turtle-magic'])

valid_wheezy = pd.get_dummies(valid_x['wheezy-copper-turtle-magic'])

test_wheezy = pd.get_dummies(test['wheezy-copper-turtle-magic'])

trn_x.drop('wheezy-copper-turtle-magic', axis=1, inplace=True)

valid_x.drop('wheezy-copper-turtle-magic', axis=1, inplace=True)

test.drop('wheezy-copper-turtle-magic', axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)

trn_x = np.concatenate([trn_x, trn_wheezy.values], axis=1)

valid_x = np.concatenate([valid_x, valid_wheezy.values], axis=1)

test = np.concatenate([test, test_wheezy.values], axis=1)
trn_x.shape, valid_x.shape, test.shape
model = build_model()

model.fit(trn_x, trn_y, epochs=80, validation_data=(valid_x, valid_y),

         callbacks=[val_loss_checkpoint, reduceLR], batch_size=512, verbose=1)
model.load_weights(weights_path)

val_preds = model.predict(valid_x, batch_size=2048, verbose=1)

roc_auc_score(valid_y.values, val_preds.reshape(-1))

for item in model.get_weights():

    print(np.sort(item))

    print("------------------------")
def build_model2():

    inp = Input(shape=(trn_x.shape[1],), name='input')

    x = layers.Dense(1000, activation='relu')(inp)

    x = layers.Dropout(0.6)(x)

    x = layers.Dense(750, activation='relu')(x)

    x = layers.Dropout(0.6)(x)

    x = layers.Dense(500, activation='relu')(x)

    x = layers.Dropout(0.6)(x)

    x = layers.Dense(1, activation='sigmoid')(x)

    

    model = Model(inp, x)

    model.compile(optimizer='adam',

                 loss='binary_crossentropy', metrics=['acc'])

    

    return model



model = build_model2()



model.fit(trn_x, trn_y, epochs=80, validation_data=(valid_x, valid_y),

         callbacks=[val_loss_checkpoint, reduceLR], batch_size=512, verbose=1)
model.load_weights(weights_path)

val_preds = model.predict(valid_x, batch_size=2048, verbose=1)

roc_auc_score(valid_y.values, val_preds.reshape(-1))

test_preds = model.predict(test, batch_size=2048, verbose=1)
sub_df = pd.read_csv(f'../input/sample_submission.csv')

sub_df.target = test_preds.reshape(-1)
sub_df.to_csv('solution.csv', index=False)

sub_df.head()