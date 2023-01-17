from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from scipy.stats import rankdata

import pandas as pd

import numpy as np

import scipy

from keras.regularizers import l1, l2, l1_l2

from keras.models import Sequential

from keras.layers import Dense

import tensorflow as tf

from keras import backend as K

import tensorflow as tf

from keras import callbacks

from keras.utils import to_categorical

SEED = 2020

N_Splits = 25

Verbose = 0

Batch_Size = 256

Epochs = 30

Regularization = l1_l2(l1=0.0, l2=0.0)
D0 = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col='id')

D_test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col='id')

y_train = D0['target']

D = D0.drop(columns='target')

test_ids = D_test.index

D_all = pd.concat([D, D_test])

num_train = len(D)

print(f'Data Shape : {D_all.shape}')        
for col in D.columns.difference(['id']):

    train_vals = set(D[col].dropna().unique())

    test_vals = set(D_test[col].dropna().unique())



    xor_cat_vals = train_vals ^ test_vals

    if xor_cat_vals:

        print(f'Replacing {len(xor_cat_vals)} values in {col}, {xor_cat_vals}')

        D_all.loc[D_all[col].isin(xor_cat_vals), col] = 'xor'
ord_maps = {

    'ord_0': {val: i for i, val in enumerate([1, 2, 3])},

    'ord_1': {

        val: i

        for i, val in enumerate(

            ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster']

        )

    },

    'ord_2': {

        val: i

        for i, val in enumerate(

            ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']

        )

    },

    'ord_3': {val: i for i, val in enumerate(sorted(D_all['ord_3'].dropna().unique()))},

    'ord_4': {val: i for i, val in enumerate(sorted(D_all['ord_4'].dropna().unique()))},

    'ord_5': {val: i for i, val in enumerate(sorted(D_all['ord_5'].dropna().unique()))},

}



ord_cols = pd.concat([D_all[col].map(ord_map).fillna(max(ord_map.values())//2).astype('float32') for col, ord_map in ord_maps.items()], axis=1)

ord_cols /= ord_cols.max() 

ord_cols_sqr = 4*(ord_cols - 0.5)**2
oh_cols = D_all.columns.difference(ord_maps.keys())

print(f'OneHot encoding {len(oh_cols)} columns')

X_oh1 = pd.get_dummies(

    D_all[oh_cols],

    columns=oh_cols,

    drop_first=True,

    dummy_na=True,

    sparse=True,

    dtype='int8',

).sparse.to_coo()
X_oh = scipy.sparse.hstack([X_oh1, ord_cols, ord_cols_sqr]).tocsr()

print(f'X_oh.shape = {X_oh.shape}')

X_train = X_oh[:num_train]

X_test = X_oh[num_train:]
def auc(y_true, y_pred):

    def fallback_auc(y_true, y_pred):

        try:

            return roc_auc_score(y_true, y_pred)

        except:

            return 0.5

    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)
oof_pred_perceptron = np.zeros((X_train.shape[0]), )

y_pred_perceptron   = np.zeros((X_test.shape[0]), )



skf = StratifiedKFold(n_splits=N_Splits, shuffle=True, random_state=SEED)



for fold, (tr_ind, val_ind) in enumerate(skf.split(X_train, y_train)):

    x_tr, x_val = X_train[tr_ind], X_train[val_ind]

    y_tr, y_val = y_train[tr_ind], y_train[val_ind]

    train_set = {'X':x_tr, 'y':to_categorical(y_tr)}

    val_set   = {'X':x_val, 'y':to_categorical(y_val)}

    model = Sequential()

    model.add(Dense(2, activation='softmax', kernel_regularizer=Regularization, input_dim=X_train.shape[1]))

    model.compile(optimizer='adam',

                  loss='categorical_crossentropy',

                  metrics=[auc],)

    es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5, verbose=Verbose, mode='max', baseline=None, restore_best_weights=True)

    sb = callbacks.ModelCheckpoint('./nn_model.w8', save_weights_only=True, save_best_only=True, verbose=Verbose)

    annealer = callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.99 ** (x+(Epochs//3)))

    model.fit(train_set['X'],

              train_set['y'],

              epochs=Epochs,

              verbose=Verbose,

              validation_data=(val_set['X'],val_set['y']),

              batch_size=Batch_Size,

              callbacks=[es, sb, annealer])

    model.load_weights('./nn_model.w8')

    fold_pred = model.predict(val_set['X'])[:,1]

    oof_pred_perceptron[val_ind] = fold_pred

    y_pred_perceptron += model.predict(X_test)[:,1] / (N_Splits)

    oof_auc_score = roc_auc_score(y_val, fold_pred)

    print(f'fold {fold+1:02} auc score is: {oof_auc_score:.6f}')

oof_auc_score = roc_auc_score(y_train, oof_pred_perceptron)

print(f'SingleLayerPerceptron OOF auc score is: {oof_auc_score}')
pd.DataFrame({'id': test_ids, 'target': y_pred_perceptron}).to_csv('submission.csv', index=False)
np.save('oof_pred_perceptron.npy', oof_pred_perceptron)

np.save('y_pred_perceptron.npy',    y_pred_perceptron)