import warnings

warnings.filterwarnings('ignore')

import os

import pandas as pd

import numpy as np

import seaborn as sns

import math

import random

import json

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import tensorflow as tf

import tensorflow_addons as tfa

import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split, KFold

from sklearn import metrics
# Basic training configurations

# Number of folds for KFold validation strategy

FOLDS = 5

# Number of epochs to train each model

EPOCHS = 130

# Batch size

BATCH_SIZE = 64

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



# Read training, test and sample submission data

train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines = True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines = True)

sample_sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')





# Target column list

target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']



# Dictionary comprehension to map the token with an specific id

token2int = {x: i for i, x in enumerate('().ACGUBEHIMSX')}



# Preprocesing function to transform features to 3d format

def preprocess_inputs(df, cols = ['sequence', 'structure', 'predicted_loop_type']):

    return np.transpose(

        np.array(

            df[cols]\

            .applymap(lambda seq: [token2int[x] for x in seq])\

            .values\

            .tolist()

        ),

        (0, 2, 1)

    )



# Transform training feature sequences to a 3d matrix of (x, 107, 3)

train_inputs = preprocess_inputs(train)

# Transform training targets sequences to a 3d matrix of (x, 68, 5)

train_labels = np.array(train[target_cols].values.tolist()).transpose(0, 2, 1)

# Get different test sets

public_test_df = test[test['seq_length'] == 107]

private_test_df = test[test['seq_length'] == 130]

# Preprocess the test sets to the same format as our training data

public_test = preprocess_inputs(public_test_df)

private_test = preprocess_inputs(private_test_df)
# Custom loss_fnc, extracted from https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183211

def CMCRMSE(y_true, y_pred):

    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)





# Function to build our wave net model

def build_model(seq_len = 107, pred_len = 68, embed_dim = 85, dropout = 0.10):

    

    def wave_block(x, filters, kernel_size, n):

        dilation_rates = [2 ** i for i in range(n)]

        x = tf.keras.layers.Conv1D(filters = filters, 

                                   kernel_size = 1,

                                   padding = 'same')(x)

        res_x = x

        for dilation_rate in dilation_rates:

            tanh_out = tf.keras.layers.Conv1D(filters = filters,

                              kernel_size = kernel_size,

                              padding = 'same', 

                              activation = 'tanh', 

                              dilation_rate = dilation_rate)(x)

            sigm_out = tf.keras.layers.Conv1D(filters = filters,

                              kernel_size = kernel_size,

                              padding = 'same',

                              activation = 'sigmoid', 

                              dilation_rate = dilation_rate)(x)

            x = tf.keras.layers.Multiply()([tanh_out, sigm_out])

            x = tf.keras.layers.Conv1D(filters = filters,

                       kernel_size = 1,

                       padding = 'same')(x)

            res_x = tf.keras.layers.Add()([res_x, x])

        return res_x

    

    inputs = tf.keras.layers.Input(shape = (seq_len, 3))

    embed = tf.keras.layers.Embedding(input_dim = len(token2int), output_dim = embed_dim)(inputs)

    reshaped = tf.reshape(embed, shape = (-1, embed.shape[1], embed.shape[2] * embed.shape[3]))

    reshaped = tf.keras.layers.SpatialDropout1D(dropout)(reshaped)

    

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, 

                                                          dropout = dropout, 

                                                          return_sequences = True, 

                                                          kernel_initializer = 'orthogonal'))(reshaped)

    x = wave_block(x, 16, 3, 12)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(dropout)(x)



    x = wave_block(x, 32, 3, 8)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(dropout)(x)



    x = wave_block(x, 64, 3, 4)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(dropout)(x)

    

    x = wave_block(x, 128, 3, 1)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(dropout)(x)

    

    

    truncated = x[:, :pred_len]

    out = tf.keras.layers.Dense(5, activation = 'linear')(truncated)

    model = tf.keras.models.Model(inputs = inputs, outputs = out)

    opt = tf.keras.optimizers.Adam(learning_rate = LR)

    opt = tfa.optimizers.SWA(opt)

    model.compile(optimizer = opt,

                  loss = tf.keras.losses.MeanSquaredLogarithmicError(),

                  metrics = [tf.keras.metrics.RootMeanSquaredError()])

    

    return model



# Evaluation metric for this problem (mean columnwise root mean squared error)

def mcrmse(y_true, y_pred):

    y_true_ = y_true.reshape(163200, 5)

    y_pred_ = y_pred.reshape(163200, 5)

    y_true0 = y_true_[:, 0]

    y_true1 = y_true_[:, 1]

    y_true2 = y_true_[:, 2]

    y_true3 = y_true_[:, 3]

    y_true4 = y_true_[:, 4]

    y_pred0 = y_pred_[:, 0]

    y_pred1 = y_pred_[:, 1]

    y_pred2 = y_pred_[:, 2]

    y_pred3 = y_pred_[:, 3]

    y_pred4 = y_pred_[:, 4]

    rmse0 = math.sqrt(metrics.mean_squared_error(y_true0, y_pred0))

    rmse1 = math.sqrt(metrics.mean_squared_error(y_true1, y_pred1))

    rmse2 = math.sqrt(metrics.mean_squared_error(y_true2, y_pred2))

    rmse3 = math.sqrt(metrics.mean_squared_error(y_true3, y_pred3))

    rmse4 = math.sqrt(metrics.mean_squared_error(y_true4, y_pred4))

    return np.mean([rmse0, rmse1, rmse2, rmse3, rmse4])





def train_and_evaluate(train_inputs, train_labels, public_test, private_test):

        

    oof_preds = np.zeros((train_inputs.shape[0], 68, 5))

    public_preds = np.zeros((public_test.shape[0], 107, 5))

    private_preds = np.zeros((private_test.shape[0], 130, 5))



    kfold = KFold(FOLDS, shuffle = True, random_state = SEED)

    for fold, (train_index, val_index) in enumerate(kfold.split(train_inputs)):

        

        print(f'Training fold {fold + 1}')

    

        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'fold_{fold + 1}.h5', 

                                                        monitor = 'val_loss',

                                                        save_best_only = True,

                                                        save_weights_only = True

                                                       )

        # Using learning rate scheduler

        cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 

                                                              mode = 'min', 

                                                              factor = 0.5, 

                                                              patience = 5, 

                                                              verbose = 1, 

                                                              min_delta = 0.00001

                                                             )

    

        x_train, x_val = train_inputs[train_index], train_inputs[val_index]

        y_train, y_val = train_labels[train_index], train_labels[val_index]

        K.clear_session()

        # Build a truncated model based on train target lengths

        model = build_model()

        history = model.fit(x_train, y_train,

                            validation_data = (x_val, y_val),

                            batch_size = BATCH_SIZE,

                            epochs = EPOCHS,

                            callbacks = [checkpoint, cb_lr_schedule],

                            verbose = VERBOSE)

    

        # Load best model to predict the validation set

        model.load_weights(f'fold_{fold + 1}.h5')

        oof_preds[val_index] = model.predict(x_val)



        # Load best model and predict the entire test sequence for the public test

        short = build_model(seq_len = 107, pred_len = 107)

        short.load_weights(f'fold_{fold + 1}.h5')

        public_preds += short.predict(public_test) / FOLDS



        # Load best model and predict the entire test sequence for the private test

        long = build_model(seq_len = 130, pred_len = 130)

        long.load_weights(f'fold_{fold + 1}.h5')

        private_preds += long.predict(private_test) / FOLDS

        

        print('-'*50)

        print('\n')

    

    # Calculate out of folds predictions

    mean_col_rmse = mcrmse(train_labels, oof_preds)



    print(f'Our out of folds mean columnwise root mean squared error is {mean_col_rmse}')

    

    return public_preds, private_preds
public_preds, private_preds = train_and_evaluate(train_inputs, train_labels, public_test, private_test)
# Function to get our predictions in the correct format

def inference_format(public_test_df, public_preds, private_test_df, private_preds, target_cols):

    predictions = []

    for test, preds in [(public_test_df, public_preds), (private_test_df, private_preds)]:

        for index, uid in enumerate(test['id']):

            single_pred = preds[index]

            single_df = pd.DataFrame(single_pred, columns = target_cols)

            # Add id

            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

            predictions.append(single_df)

            

    predictions = pd.concat(predictions)

    return predictions



# Get our predictions in the correct format

predictions = inference_format(public_test_df, public_preds, private_test_df, private_preds, target_cols)

# Sanity check

submission = sample_sub[['id_seqpos']].merge(predictions, on = ['id_seqpos'])

submission.to_csv('submission.csv', index = False)

print('Submission saved')

submission.head()