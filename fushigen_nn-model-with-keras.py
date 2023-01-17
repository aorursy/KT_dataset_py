import numpy as np 

import pandas as pd 

import os



from sklearn.model_selection import KFold

from sklearn.metrics import log_loss

from sklearn.decomposition import PCA



import tensorflow as tf

import tensorflow_addons as tfa

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

import time
import sys

sys.path.append('../input/stratified')

from ml_stratifiers import MultilabelStratifiedKFold
test_df = pd.read_csv('../input/lish-moa/test_features.csv')

train_df = pd.read_csv('../input/lish-moa/train_features.csv')

train_target_df = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

sub = pd.read_csv('../input/lish-moa/sample_submission.csv')



target_cols = train_target_df.columns[1:]

N_TARGETS = len(target_cols)

print(train_df.shape)
GENES = [col for col in train_df.columns if col.startswith('g-')]

CELLS = [col for col in train_df.columns if col.startswith('c-')]
# GENES

n_comp = 50



data = pd.concat([pd.DataFrame(train_df[GENES]), pd.DataFrame(test_df[GENES])])

data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[GENES]))

train2 = data2[:train_df.shape[0]]; test2 = data2[-test_df.shape[0]:]



train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])

test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])



# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]

train_df = pd.concat((train_df, train2), axis=1)

test_df = pd.concat((test_df, test2), axis=1)
train_df.shape
#CELLS

n_comp = 15



data = pd.concat([pd.DataFrame(train_df[CELLS]), pd.DataFrame(test_df[CELLS])])

data2 = (PCA(n_components=n_comp, random_state=42).fit_transform(data[CELLS]))

train2 = data2[:train_df.shape[0]]; test2 = data2[-test_df.shape[0]:]



train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])

test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])



# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]

train_df = pd.concat((train_df, train2), axis=1)

test_df = pd.concat((test_df, test2), axis=1)
train_df.shape
from sklearn.feature_selection import VarianceThreshold



train_copy = train_df

var_thresh = VarianceThreshold(0.8)  #<-- Update

data = train_df.append(test_df)

data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

#arrayで出力される

#どういう特徴量が選ばれたかわからない



train_df_transformed = data_transformed[ : train_df.shape[0]]

test_df_transformed = data_transformed[-test_df.shape[0] : ]





train_df = pd.DataFrame(train_df[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\

                              columns=['sig_id','cp_type','cp_time','cp_dose'])



train_df = pd.concat([train_df, pd.DataFrame(train_df_transformed)], axis=1)





test_df = pd.DataFrame(test_df[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\

                             columns=['sig_id','cp_type','cp_time','cp_dose'])



test_df = pd.concat([test_df, pd.DataFrame(test_df_transformed)], axis=1)



train_df.shape
#num_dict = {}

#for i in np.arange(0,868):

#    num_dict[i] = f'{i}'

#train_df = train_df.rename(columns=num_dict)

#test_df = test_df.rename(columns=num_dict)
search_row = dict(train_copy.iloc[0,4:])

col_rela = {}

for i in np.arange(0,868):

    for k, v in search_row.items():

        if train_df[i][0] == v:

            col_rela[i] = k

train_df = train_df.rename(columns=col_rela)

test_df = test_df.rename(columns=col_rela)
train_df
SEED = 1234

EPOCHS = 28

BATCH_SIZE = 128

FOLDS = 5

REPEATS = 5

LR = 0.0005

N_TARGETS = len(target_cols)
def seed_everything(seed):

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)
def multi_log_loss(y_true, y_pred):

    losses = []

    for col in y_true.columns:

        losses.append(log_loss(y_true.loc[:, col], y_pred.loc[:, col]))

    return np.mean(losses)
def preprocess_df(data):

    data['cp_type'] = (data['cp_type'] == 'trt_cp').astype(int)

    data['cp_dose'] = (data['cp_dose'] == 'D2').astype(int)

    return data
x_train = preprocess_df(train_df.drop(columns="sig_id"))

x_test =preprocess_df(test_df.drop(columns="sig_id"))

y_train = train_target_df.drop(columns="sig_id")

N_FEATURES = x_train.shape[1]
#x_train = np.asarray(x_train)

#x_test = np.asarray(x_test)

#y_train = np.asarray(y_train)
#VarianceThersholdの時は必要

x_train = x_train.astype({'cp_time':int})

x_test = x_test.astype({'cp_time':int})
def create_model():

    model = tf.keras.Sequential([

    tf.keras.layers.Input(N_FEATURES),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.2),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),

    tf.keras.layers.BatchNormalization(),

    #tf.keras.layers.Dropout(0.4),

    #tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),  

    #tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(N_TARGETS, activation="sigmoid"))

    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = LR), loss='binary_crossentropy', metrics=["accuracy"])

    return model
def build_train(resume_models = None, repeat_number = 0, folds = 5, skip_folds = 0):

    

    models = []

    oof_preds = y_train.copy()

    



    kfold = KFold(folds, shuffle = True)

    #kfold = MultilabelStratifiedKFold(n_splits=folds)

    # stratifiedの時はX=x_train,y=y_train

    for fold, (train_ind, val_ind) in enumerate(kfold.split(x_train)):

        print('\n')

        print('-'*50)

        print(f'Training fold {fold + 1}')

        

        cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'binary_crossentropy', factor = 0.4, patience = 2, verbose = 1, min_delta = 0.0001, mode = 'auto')

        checkpoint_path = f'repeat:{repeat_number}_Fold:{fold}.hdf5'

        cb_checkpt = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 0, save_best_only = True, save_weights_only = True, mode = 'min')



        model = create_model()

        model.fit(x_train.values[train_ind],

              y_train.values[train_ind],

              validation_data=(x_train.values[val_ind], y_train.values[val_ind]),

              callbacks = [cb_lr_schedule, cb_checkpt],

              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2

             )

        model.load_weights(checkpoint_path)

        oof_preds.loc[val_ind, :] = model.predict(x_train.values[val_ind])

        models.append(model)

        print('train:')

        print(list(zip(model.metrics_names, model.evaluate(x_train.values[train_ind], y_train.values[train_ind], verbose=0, batch_size=32))))

        print('val:')

        print(list(zip(model.metrics_names, model.evaluate(x_train.values[val_ind], y_train.values[val_ind], verbose=0, batch_size=32))))



    return models, oof_preds
model = create_model()

model.summary()
start = time.time()

models = []

oof_preds = []

# seed everything

seed_everything(SEED)

for i in range(REPEATS):

    m, oof = build_train(repeat_number = i, folds=FOLDS)

    models = models + m

    oof_preds.append(oof)

#一回専用

#m, oof = build_train(repeat_number = i, folds=FOLDS)

#models = models + m

#oof_preds.append(oof)



finish = time.time()-start

print(finish)
models[1].predict(x_test)
test_preds = sub.copy()

test_preds[target_cols] = 0

for model in models:

    test_preds.loc[:,target_cols] += model.predict(x_test)

test_preds.loc[:,target_cols] /= len(models)

test_preds.loc[x_test['cp_type'] == 0, target_cols] = 0

test_preds.to_csv('submission.csv', index=False)