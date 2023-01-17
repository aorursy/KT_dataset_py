import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from keras import layers, Input, Model

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



%matplotlib inline
data_dir = '../input/'

!ls {data_dir}
train_raw = pd.read_csv(f'{data_dir}train.csv')

train_raw.head()
test_raw = pd.read_csv(f'{data_dir}test.csv')

test_raw.head()
train_raw.shape, test_raw.shape
train_raw.isnull().sum().sum(), test_raw.isnull().sum().sum()
sns.countplot(train_raw.target)

plt.show()
train_raw.target.value_counts()
trn_x, valid_x, trn_y, valid_y = train_test_split(train_raw.drop(['id', 'target'], axis=1), train_raw.target, random_state=33, test_size=0.15)

trn_x.shape, valid_x.shape, trn_y.shape, valid_y.shape
trn_wheezy = pd.get_dummies(trn_x['wheezy-copper-turtle-magic'])

valid_wheezy = pd.get_dummies(valid_x['wheezy-copper-turtle-magic'])

test_wheezy = pd.get_dummies(test_raw['wheezy-copper-turtle-magic'])



trn_wheezy.shape, valid_wheezy.shape, test_wheezy.shape
trn_x.drop('wheezy-copper-turtle-magic', axis=1, inplace=True)

valid_x.drop('wheezy-copper-turtle-magic', axis=1, inplace=True)

test_raw.drop('wheezy-copper-turtle-magic', axis=1, inplace=True)
sc = StandardScaler()

trn_x = sc.fit_transform(trn_x)



valid_x = sc.transform(valid_x)

test_x = sc.transform(test_raw.drop('id', axis=1))
trn_x = np.concatenate([trn_x, trn_wheezy.values], axis=1)

valid_x = np.concatenate([valid_x, valid_wheezy.values], axis=1)

test_x = np.concatenate([test_x, test_wheezy.values], axis=1)
def build_model():

    inp = Input(shape=(trn_x.shape[1],), name='input')

    x = layers.Dense(1000, activation='relu')(inp)

    x = layers.Dropout(0.65)(x)

    x = layers.Dense(750, activation='relu')(x)

    x = layers.Dropout(0.65)(x)

    x = layers.Dense(500, activation='relu')(x)

    x = layers.Dropout(0.6)(x)

    x = layers.Dense(1, activation='sigmoid')(x)

    

    model = Model(inp, x)

    model.compile(optimizer='adam',

                 loss='binary_crossentropy', metrics=['acc'])

    

    return model



model = build_model()

model.summary()
weights_path = f'weights.best.hdf5'

val_loss_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, mode='min', min_lr=1e-6)
model.fit(trn_x, trn_y, epochs=80, validation_data=(valid_x, valid_y),

         callbacks=[val_loss_checkpoint, reduceLR], batch_size=512, verbose=1)
model.load_weights(weights_path)
val_preds = model.predict(valid_x, batch_size=2048, verbose=1)
roc_auc_score(valid_y.values, val_preds.reshape(-1))
test_preds = model.predict(test_x, batch_size=2048, verbose=1)
sub_df = pd.read_csv(f'{data_dir}sample_submission.csv')

sub_df.target = test_preds.reshape(-1)

sub_df.head()
sub_df.to_csv('solution.csv', index=False)