

import numpy as np 

import pandas as pd 

import os



from sklearn.model_selection import KFold

from sklearn.metrics import log_loss

 

import tensorflow as tf

import tensorflow_addons as tfa

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M
test_df = pd.read_csv('../input/lish-moa/test_features.csv')

train_df = pd.read_csv('../input/lish-moa/train_features.csv')

train_target_df = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

sub = pd.read_csv('../input/lish-moa/sample_submission.csv')



target_cols = train_target_df.columns[1:]

N_TARGETS = len(target_cols)
SEED = 2020

EPOCHS = 30

BATCH_SIZE = 128

FOLDS = 5

REPEATS = 5

LR = 0.00045

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
def create_model():

    model = tf.keras.Sequential([

    tf.keras.layers.Input(N_FEATURES),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.3),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(1024, activation="relu")),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.4),

    #tfa.layers.WeightNormalization(tf.keras.layers.Dense(1024, activation="relu")),

    #tf.keras.layers.BatchNormalization(),

    #tf.keras.layers.Dropout(0.4),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(1024, activation="relu")),  

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.3),

    tfa.layers.WeightNormalization(tf.keras.layers.Dense(N_TARGETS, activation="sigmoid"))

    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = LR), loss='binary_crossentropy', metrics=["accuracy"])

    return model
def build_train(resume_models = None, repeat_number = 0, folds = 7, skip_folds = 0):

    

    models = []

    oof_preds = y_train.copy()

    



    kfold = KFold(folds, shuffle = True)

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



    return models, oof_preds
models = []

oof_preds = []

# seed everything

seed_everything(SEED)

for i in range(REPEATS):

    m, oof = build_train(repeat_number = i, folds=FOLDS)

    models = models + m

    oof_preds.append(oof)



mean_oof_preds = y_train.copy()

mean_oof_preds.loc[:, target_cols] = 0

for i, p in enumerate(oof_preds):

    print(f"Repeat {i + 1} OOF Log Loss: {multi_log_loss(y_train, p)}")

    mean_oof_preds.loc[:, target_cols] += p[target_cols]



mean_oof_preds.loc[:, target_cols] /= len(oof_preds)

print(f"Mean OOF Log Loss: {multi_log_loss(y_train, mean_oof_preds)}")

mean_oof_preds.loc[x_train['cp_type'] == 0, target_cols] = 0

print(f"Mean OOF Log Loss (ctl adjusted): {multi_log_loss(y_train, mean_oof_preds)}")
test_preds = sub.copy()

test_preds[target_cols] = 0

for model in models:

    test_preds.loc[:,target_cols] += model.predict(x_test)

test_preds.loc[:,target_cols] /= len(models)

test_preds.loc[x_test['cp_type'] == 0, target_cols] = 0

test_preds.to_csv('submission.csv', index=False)