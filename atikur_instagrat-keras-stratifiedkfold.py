import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from keras import layers, Input, Model

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold



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
trn_wheezy = pd.get_dummies(train_raw['wheezy-copper-turtle-magic'])

test_wheezy = pd.get_dummies(test_raw['wheezy-copper-turtle-magic'])



trn_wheezy.shape, test_wheezy.shape
target = train_raw.target



train_raw.drop(['id', 'wheezy-copper-turtle-magic', 'target'], axis=1, inplace=True)

test_raw.drop(['id', 'wheezy-copper-turtle-magic'], axis=1, inplace=True)
sc = StandardScaler()

train_x = sc.fit_transform(train_raw)

test_x = sc.transform(test_raw)
train_x = np.concatenate([train_x, trn_wheezy.values], axis=1)

test_x = np.concatenate([test_x, test_wheezy.values], axis=1)
def build_model():

    inp = Input(shape=(train_x.shape[1],), name='input')

    x = layers.Dense(1500, activation='relu')(inp)

    x = layers.Dropout(0.6)(x)

    x = layers.Dense(1000, activation='relu')(x)

    x = layers.Dropout(0.55)(x)

    x = layers.Dense(500, activation='relu')(x)

    x = layers.Dropout(0.55)(x)

    x = layers.Dense(1, activation='sigmoid')(x)

    

    model = Model(inp, x)

    model.compile(optimizer='adam',

                 loss='binary_crossentropy', metrics=['acc'])

    

    return model



model = build_model()

model.summary()
NFOLDS = 20

NEPOCHS = 75

folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)



oof = np.zeros(train_raw.shape[0])

predictions = np.zeros(test_raw.shape[0])



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, target)):

    print(f'Fold - {fold_ + 1}')

    

    weights_path = f'weights.best.hdf5'

    val_loss_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

    reduceLR = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=0, mode='min', min_lr=1e-6)

    

    model = build_model()

    model.fit(train_x[trn_idx], target.values[trn_idx], epochs=NEPOCHS, validation_data=(train_x[val_idx], target.values[val_idx]),

         callbacks=[val_loss_checkpoint, reduceLR], batch_size=512, verbose=0)

    model.load_weights(weights_path)

    

    val_preds = model.predict(train_x[val_idx], batch_size=2048, verbose=0)

    print(f'ROC AUC: {roc_auc_score(target.values[val_idx], val_preds.reshape(-1))}')

    

    test_preds = model.predict(test_x, batch_size=2048, verbose=0)

        

    oof[val_idx] = val_preds.reshape(-1)

    predictions += test_preds.reshape(-1)/folds.n_splits
roc_auc_score(target.values, oof)
sub_df = pd.read_csv(f'{data_dir}sample_submission.csv')

sub_df.target = predictions

sub_df.head()
sub_df.to_csv('solution.csv', index=False)