import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow.keras.backend as K

import tensorflow_addons as tfa

from sklearn.metrics import log_loss

from tqdm.notebook import tqdm

import random

import os

from sklearn.preprocessing import StandardScaler
# Basic training configurations

# Number of folds for KFold validation strategy

FOLDS = 5

# Number of epochs to train each model

EPOCHS = 80

# Batch size

BATCH_SIZE = 128

# Learning rate

LR = 0.001

# Verbosity

VERBOSE = 2

# Seed for deterministic results

SEED = 123



# Function to seed everything

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)

    

seed_everything(SEED)
def mapping_and_filter(train, train_targets, test):

    cp_type = {'trt_cp': 0, 'ctl_vehicle': 1}

    cp_dose = {'D1': 0, 'D2': 1}

    for df in [train, test]:

        df['cp_type'] = df['cp_type'].map(cp_type)

        df['cp_dose'] = df['cp_dose'].map(cp_dose)

    train_targets = train_targets[train['cp_type'] == 0].reset_index(drop = True)

    train = train[train['cp_type'] == 0].reset_index(drop = True)

    train_targets.drop(['sig_id'], inplace = True, axis = 1)

    return train, train_targets, test





def scaling(train, test):

    features = train.columns[2:]

    scaler = StandardScaler()

    train[features] = scaler.fit_transform(train[features])

    test[features] = scaler.transform(test[features])

    return train, test, features





# Function to calculate the mean los loss of the targets

def mean_log_loss(y_true, y_pred):

    metrics = []

    for target in range(len(train_targets.columns)):

        metrics.append(log_loss(y_true[:, target], y_pred[:, target]))

    return np.mean(metrics)



# Function to create our dnn

def create_model(shape):

    inp = tf.keras.layers.Input(shape = (shape))

    x = tf.keras.layers.BatchNormalization()(inp)

    x = tf.keras.layers.Dropout(0.2)(x)

    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation = 'relu'))(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(1048, activation = 'relu'))(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.5)(x)

    out = tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation = 'sigmoid'))(x)

    model = tf.keras.models.Model(inputs = inp, outputs = out)

    opt = tf.optimizers.Adam()

    opt = tfa.optimizers.Lookahead(opt, sync_period = 10)

    model.compile(optimizer = opt, 

                  loss = 'binary_crossentropy')

    return model





# Function to train our dnn

def train_and_evaluate(train, train_targets, test, features, perm_imp = False):

    models = []

    trn_indices, val_indices = [], []

    oof_pred = np.zeros((train.shape[0], 206))

    test_pred = np.zeros((test.shape[0], 206))

    for fold, (trn_ind, val_ind) in enumerate(MultilabelStratifiedKFold(n_splits = FOLDS, 

                                                                        random_state = SEED, 

                                                                        shuffle = True)\

                                              .split(train_targets, train_targets)):

        print(f'Training fold {fold + 1}')

        K.clear_session()

        model = create_model(len(features))

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',

                                                          mode = 'min',

                                                          patience = 10,

                                                          restore_best_weights = True,

                                                          verbose = 1)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',

                                                         mode = 'min',

                                                         factor = 0.3,

                                                         patience = 3)



        x_train, x_val = train[features].values[trn_ind], train[features].values[val_ind]

        y_train, y_val = train_targets.values[trn_ind], train_targets.values[val_ind]



        model.fit(x_train, y_train,

                  validation_data = (x_val, y_val),

                  epochs = EPOCHS, 

                  batch_size = BATCH_SIZE,

                  callbacks = [early_stopping, reduce_lr],

                  verbose = VERBOSE)



        oof_pred[val_ind] = model.predict(x_val)

        test_pred += model.predict(test[features].values) / FOLDS

        

        models.append(model)

        trn_indices.append(trn_ind)

        val_indices.append(val_ind)

        

        print('-'*50)

        print('\n')





    oof_score = mean_log_loss(train_targets.values, oof_pred)

    print(f'Our out of folds mean log loss score is {oof_score}')

    

    if perm_imp:

        return models, trn_indices, val_indices, oof_score

    else:

        return test_pred

    

    

# Function to perform permutation importance (feature selection)

def permutation_importance(train, train_targets, test, features):

    # Get the base score of our model

    models, trn_indices, val_indices, oof_score = train_and_evaluate(train, train_targets, test, features, perm_imp = True)

    scores = np.zeros(len(features))

    # Iterate over each feature

    for num, feature in enumerate(tqdm(features)):

        train_ = train.copy()

        # Shuffle the data for the given feature

        train_[feature] = train_[feature].sample(frac = 1.0).reset_index(drop = True)

        oof_pred = np.zeros((train_.shape[0], 206))

        # Predict each validation fold with the corresponding model

        for model, trn_ind, val_ind in zip(models, trn_indices, val_indices):

            oof_pred[val_ind] = model.predict(train_[features].values[val_ind])

        score = mean_log_loss(train_targets.values, oof_pred)

        # If the result is positive the feature is useless, if it is negative is usefull

        scores[num] = oof_score - score

        

    perm_dataframe = pd.DataFrame({'features': features, 'score': scores})

    perm_dataframe = perm_dataframe[perm_dataframe['score'] < 0]

    new_features = list(perm_dataframe['features'])

    print(f'We have select {len(new_features)} features')

    return new_features
train = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test = pd.read_csv('../input/lish-moa/test_features.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')



train, train_targets, test = mapping_and_filter(train, train_targets, test)



train, test, features = scaling(train, test)



new_features = permutation_importance(train, train_targets, test, features)
test_preds = train_and_evaluate(train, train_targets, test, new_features, perm_imp = False)

sample_submission.loc[:, train_targets.columns] = test_preds

sample_submission.loc[test['cp_type'] == 1, train_targets.columns] = 0

sample_submission.to_csv('submission.csv', index = False)

sample_submission.head()