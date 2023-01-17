import pandas as pd

import numpy as np

import os

import pickle



from sklearn.model_selection import KFold

from sklearn.metrics import log_loss



import tensorflow as tf
train_df = pd.read_csv('../input/lish-moa/train_features.csv')

test_df = pd.read_csv('../input/lish-moa/test_features.csv')



train_target_df = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

sample_sub = pd.read_csv('../input/lish-moa/sample_submission.csv')
train_df.head()
train_target_df.head()
# we don't need sig_id as our target col

target_cols = train_target_df.columns[1:]

N_TARGETS = len(target_cols)
def seed_everything(seed):

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)
sample_sub.head()
# multi log loss function

def multi_log_loss(y_true, y_pred):

    losses = []

    for col in y_true.columns:

        losses.append(log_loss(y_true.loc[:, col], y_pred.loc[:, col]))

    return np.mean(losses)
# pre-processing

def clean_df(data):

    data['cp_type'] = (data['cp_type'] == 'trt_cp').astype(int)

    data['cp_dose'] = (data['cp_dose'] == 'D2').astype(int)

    return data
X_train = clean_df(train_df.drop(["sig_id"], axis=1))

X_test = clean_df(test_df.drop(['sig_id'], axis=1))

y_train = train_target_df.drop(['sig_id'], axis=1)

N_FEATURES = X_train.shape[1]
# basic setup

SEED = 1234

EPOCHS = 28

BATCH_SIZE = 128

FOLDS = 5

REPEATS = 5

LR = 0.0005

N_TARGETS = len(target_cols)
def build_model(n_hidden=3, n_neurons=10, learning_rate=3e-3, input_shape=N_FEATURES, activation="relu", optimizer="adam"):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))



    for layer in range(n_hidden):

        if layer == 1:

            model.add(tf.keras.layers.Dropout(0.2))

        elif layer == 2:

            model.add(tf.keras.layers.Dropout(0.5))

        if activation == "selu":

            model.add(tf.keras.layers.Dense(n_neurons, activation = "selu", kernel_initializer="lecun_normal"))

        elif activation == "elu":

            model.add(tf.keras.layers.Dense(n_neurons, activation = "elu", kernel_initializer = "he_normal", kernel_regularizer = tf.keras.regularizers.l2(0.01)))

        else:

            model.add(tf.keras.layers.Dense(n_neurons, activation = activation))





    model.add(tf.keras.layers.Dense(N_TARGETS, activation = "sigmoid"))

    if optimizer == "sgd":

        optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model



early_stop = tf.keras.callbacks.EarlyStopping(patience = 3, restore_best_weights = True)
def build_train(resume_models = None, repeat_number = 0, folds = 5, skip_folds = 0):

    

    models = []

    oof_preds = y_train.copy()

    



    kfold = KFold(folds, shuffle = True)

    for fold, (train_ind, val_ind) in enumerate(kfold.split(X_train)):

        print('\n')

        print('-'*50)

        print(f'Training fold {fold + 1}')

        

        cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'binary_crossentropy', factor = 0.4, patience = 2, verbose = 1, min_delta = 0.0001, mode = 'auto')

        checkpoint_path = f'repeat:{repeat_number}_Fold:{fold}.hdf5'

        cb_checkpt = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 0, save_best_only = True, save_weights_only = True, mode = 'min')



        model = build_model(n_hidden=4, learning_rate=0.001, n_neurons=200, optimizer="adam", activation="relu")

        model.fit(X_train.values[train_ind],

              y_train.values[train_ind],

              validation_data=(X_train.values[val_ind], y_train.values[val_ind]),

              callbacks = [cb_lr_schedule, cb_checkpt],

              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2

             )

        model.load_weights(checkpoint_path)

        oof_preds.loc[val_ind, :] = model.predict(X_train.values[val_ind])

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

mean_oof_preds.loc[X_train['cp_type'] == 0, target_cols] = 0

print(f"Mean OOF Log Loss (ctl adjusted): {multi_log_loss(y_train, mean_oof_preds)}")
test_preds = sample_sub.copy()

test_preds[target_cols] = 0

for model in models:

    test_preds.loc[:,target_cols] += model.predict(X_test)

test_preds.loc[:,target_cols] /= len(models)

test_preds.loc[X_test['cp_type'] == 0, target_cols] = 0

test_preds.to_csv('submission.csv', index=False)
test_preds.head()