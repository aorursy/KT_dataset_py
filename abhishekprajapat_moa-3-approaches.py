!pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import numpy as np

import pandas as pd

import tensorflow as tf



from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import tensorflow_addons as tfa



from sklearn.metrics import log_loss

from tqdm.notebook import tqdm



import warnings



warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_non_targets = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')



test_features = pd.read_csv('../input/lish-moa/test_features.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')
train_features.head(1)
train_non_targets.head(1)
train_non_targets.head(1)
values = np.sum(train_non_targets.iloc[:, 1:], axis = 0)



sns.distplot(values)
def create_fold(data):

    

    data['fold'] = -1

    

    data = data.sample(frac = 1).reset_index(drop = True)

    

    targets = data.drop('sig_id', axis=1).values

    

    splitter = MultilabelStratifiedKFold(n_splits=7, random_state=0)

    

    for fold, (train, valid) in enumerate(splitter.split(X=data, y=targets)):

        

        data.loc[valid, 'fold'] = fold

        

    return data
combined = train_targets.merge(train_non_targets, on='sig_id', how='outer')

combined = create_fold(combined)

train_targets['fold'] = combined['fold']

train_non_targets['fold'] = combined['fold']



del(combined)
train_targets.to_csv('fold_data_targets.csv', index = False)
train_non_targets.to_csv('fold_data_non_targets.csv', index = False)
def preprocess(data):

    

    cp_time = pd.get_dummies(data['cp_time'])

    cp_type = pd.get_dummies(data['cp_type'])

    cp_dose = pd.get_dummies(data['cp_dose'])

    

    

    data = data.join(cp_time)

    data = data.join(cp_type)

    data = data.join(cp_dose)

    

    data.drop(columns = ['cp_time', 'cp_dose', 'cp_type'], inplace=True)

    

    return data
train_features = preprocess(train_features)
def uni_non_targets():

    train_df = train_features.merge(train_non_targets, on='sig_id', how='outer')   

    return train_df
def create_model(num_inputs, num_outputs):

    

    model = tf.keras.Sequential([

        

        tf.keras.layers.Input(num_inputs),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.2),

        

        tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048, activation="relu")),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.5),

        

        tfa.layers.WeightNormalization(tf.keras.layers.Dense(1048, activation="relu")),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.5),

        

        tfa.layers.WeightNormalization(tf.keras.layers.Dense(num_outputs, activation="sigmoid"))

        

    ])

    

    

    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period=10),

                  loss='binary_crossentropy', 

                  )

    

    return model
def metric(y_true, y_pred):

    

    metrics = []

    

    for _target in train_targets.columns[1:-1]:

        

        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels=[0,1]))

        

    return np.mean(metrics)
train_df = train_features.merge(train_targets, on='sig_id', how='outer')
def run_model(fold):

    

    train_df = train_features.merge(train_targets, on='sig_id', how='outer')

    

    # defining the parameters

    cols = train_df.columns

    

    ID = cols[0]

    fold_col = cols[-1]

    features = cols[1:880]

    # we are skipping the mid (947th column) as it is the fold column of the previous part

    targets = cols[880:-1]

    

    # loading the data

    train = train_df[train_df['fold'] != fold]

    valid = train_df[train_df['fold'] == fold]

    

    x_train = train.loc[:, features]

    x_valid = valid.loc[:, features]

    

    y_train = train.loc[:, targets]

    y_valid = valid.loc[:, targets]

    

    # creating the model

    model = create_model(x_train.shape[1], y_train.shape[1])

    

    # Defining model callbacks

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')

    

    checkpoint_path = f'Fold_{fold}_basic.hdf5'

    

    cb_checkpt = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True, 

                                save_weights_only=True, mode='min')

    

    # Fitting the model

    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=35, batch_size=128,

             callbacks = [reduce_lr_loss, cb_checkpt], verbose=1)

    

    # Loading the best model

    model.load_weights(checkpoint_path)

    

    # Making the predictions

    y_valid_pred = model.predict(x_valid)

    

    # converting the predictions to dataframe

    y_valid_pred = pd.DataFrame(y_valid_pred, columns=y_valid.columns)

    

    # Evaluating the final results

    print('\n\n\n')

    print('OOF Metric: ', metric(y_valid, y_valid_pred))

    

    return
run_model(0)
def run_part1(fold):

    

    # loading the targets_non_scored concatinated data

    train_df = uni_non_targets()

    

    # defining the parameters

    cols = train_df.columns

    

    ID = cols[0]

    fold_col = cols[-1]

    features = cols[1:880]

    targets = cols[880:-1]

    

    # loading the data

    train = train_df[train_df['fold'] != fold]

    valid = train_df[train_df['fold'] == fold]

    

    x_train = train.loc[:, features]

    x_valid = valid.loc[:, features]

    

    y_train = train.loc[:, targets]

    y_valid = valid.loc[:, targets]

    

    # Printing the shape

    print(x_train.shape[1], y_train.shape[1])

    

    # creating the model

    model = create_model(x_train.shape[1], y_train.shape[1])

    

    # Defining model callbacks

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')

    

    checkpoint_path = f'Fold_{fold}_part1.hdf5'

    

    cb_checkpt = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True, 

                                save_weights_only=True, mode='min')

    

    # Fitting the model

    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=35, batch_size=128,

             callbacks = [reduce_lr_loss, cb_checkpt], verbose=1)

    

    # Loading the best weights model

    model.load_weights(checkpoint_path)

    

    # Making the predictions

    y_valid_pred = model.predict(x_valid)

    y_train_pred = model.predict(x_train)

    

    # converting the y_preds

    y_valid_pred = pd.DataFrame(y_valid_pred, columns = y_valid.columns)

    y_train_pred = pd.DataFrame(y_train_pred, columns = y_valid.columns)

    

    # replacing the train_df with the predicted data

    train_df.loc[:, targets][train_df['fold'] != fold] = y_train_pred

    train_df.loc[:, targets][train_df['fold'] == fold] = y_valid_pred

    

    # drop this fold

    train_df.drop(columns='fold', inplace=True)

    

    return train_df
def run_part2(fold):

    

    # Prepairing final data

    features = run_part1(fold)

    targets = train_targets

    

    # Merging both

    train_df = features.merge(targets, on='sig_id', how='outer')

    

    # defining the parameters

    cols = train_df.columns

    

    ID = cols[0]

    fold_col = cols[-1]

    features = cols[1:1282]

    targets = cols[1282:-1]

    

    # loading the data

    train = train_df[train_df['fold'] != fold]

    valid = train_df[train_df['fold'] == fold]

    

    x_train = train.loc[:, features]

    x_valid = valid.loc[:, features]

    

    y_train = train.loc[:, targets]

    y_valid = valid.loc[:, targets]

    

    # Some blank lines

    print('\n\n\n')

    

    # Printing the input shape

    print(x_train.shape[1], y_train.shape[1])

    

    # creating the model

    model = create_model(x_train.shape[1], y_train.shape[1])

    

    # Defining model callbacks

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')

    

    checkpoint_path = f'Fold_{fold}_part2.hdf5'

    

    cb_checkpt = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True, 

                                save_weights_only=True, mode='min')

    

    

    # Fitting the model

    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=35, batch_size=128,

             callbacks = [reduce_lr_loss, cb_checkpt], verbose=1)

    

    # Loading the best model

    model.load_weights(checkpoint_path)

    

    # Making the predictions

    y_valid_pred = model.predict(x_valid)

    

    # converting the predictions to dataframe

    y_valid_pred = pd.DataFrame(y_valid_pred, columns=y_valid.columns.values)

    

    # Evaluating the final results

    print('\n\n\n')

    print('OOF Metric: ', metric(y_valid, y_valid_pred))

    

    return 
def run(fold):

    

    run_part2(fold)

    

    return
run(0)
def Seq_model(num_inputs, num_outputs):

    

    model = tf.keras.Sequential([

        

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024), input_shape=(1, num_inputs)),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.5),

        

        tfa.layers.WeightNormalization(tf.keras.layers.Dense(1048, activation="relu")),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.5),

        

        tfa.layers.WeightNormalization(tf.keras.layers.Dense(num_outputs, activation="sigmoid"))

        

    ])

    

    

    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(), sync_period=10),

                  loss='binary_crossentropy', 

                  )

    

    return model
def run_seq_model(fold):

    

    train_df = train_features.merge(train_targets, on='sig_id', how='outer')

    

    # defining the parameters

    cols = train_df.columns

    

    ID = cols[0]

    fold_col = cols[-1]

    features = cols[1:880]

    # we are skipping the mid (947th column) as it is the fold column of the previous part

    targets = cols[880:-1]

    

    # loading the data

    train = train_df[train_df['fold'] != fold]

    valid = train_df[train_df['fold'] == fold]

    

    x_train = train.loc[:, features]

    x_valid = valid.loc[:, features]

    

    y_train = train.loc[:, targets]

    y_valid = valid.loc[:, targets]

    

    

    # reshaping the data for LSTM

    x_train = np.array(x_train).reshape(-1, 1, 879)

    x_valid = np.array(x_valid).reshape(-1, 1, 879)

    

    # Printing the input shape

    print(x_train.shape[2], y_train.shape[1])

    

    # creating the model

    model = Seq_model(x_train.shape[2], y_train.shape[1])

    

    # Defining model callbacks

    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')

    

    checkpoint_path = f'Fold_{fold}_Seq.hdf5'

    

    cb_checkpt = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True, 

                                save_weights_only=True, mode='min')

    

    # Fitting the model

    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=35, batch_size=128,

             callbacks = [reduce_lr_loss, cb_checkpt], verbose=1)

    

    # Loading the best model

    model.load_weights(checkpoint_path)

    

    # Making the predictions

    y_valid_pred = model.predict(x_valid)

    

    # converting the predictions to dataframe

    y_valid_pred = pd.DataFrame(y_valid_pred, columns=y_valid.columns)

    

    # Evaluating the final results

    print('\n\n\n')

    print('OOF Metric: ', metric(y_valid, y_valid_pred))

    

    return
run_seq_model(0)
sample_submission.head(2)
test_features.head(2)
test_features = preprocess(test_features)
test_features.head(2)
model = create_model(879, 206)



model.load_weights('Fold_0_basic.hdf5')



pred = model.predict(test_features.iloc[:, 1:])
pred = pd.DataFrame(pred, columns = train_targets.columns.values[1:-1])



sub_file1 = sample_submission.copy()

sub_file1.iloc[:, 1:] = pred
sub_file1.to_csv('sub1.csv', index=False)
model_part1 = create_model(879, 402)



model_part1.load_weights('Fold_0_part1.hdf5')



pred_1 = model_part1.predict(test_features.iloc[:, 1:])



pred_1 = pd.DataFrame(pred_1, columns = train_non_targets.columns.values[1:-1])

pred_1['sig_id'] = test_features['sig_id']



features_test = test_features.copy()



features_test = features_test.merge(pred_1, on='sig_id', how='outer')
model_part2 = create_model(1281, 206)



model_part2.load_weights('./Fold_0_part2.hdf5')



pred_2 = model_part2.predict(features_test.iloc[:, 1:])

pred_2 = pd.DataFrame(pred_2, columns = train_targets.columns.values[1:-1])



sub_file2 = sample_submission.copy()

sub_file2.iloc[:, 1:] = pred_2
sub_file2.to_csv('sub2.csv', index=False)
model_seq = Seq_model(879, 206)



model_seq.load_weights('./Fold_0_Seq.hdf5')



to_pred = np.array(test_features.iloc[:, 1:]).reshape(-1, 1, 879)



pred_3 = model_seq.predict(to_pred)



pred_3 = pd.DataFrame(pred, columns = train_targets.columns.values[1:])



sub_file3 = sample_submission.copy()

sub_file3.iloc[:, 1:] = pred_3
sub_file3.to_csv('sub3.csv', index=False)