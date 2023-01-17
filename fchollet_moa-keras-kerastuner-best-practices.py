!pip install git+https://github.com/keras-team/keras-tuner.git -q
import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



print('TF version:', tf.__version__)

print('GPU devices:', tf.config.list_physical_devices('GPU'))
train_features_df = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets_df = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

test_features_df = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



print('train_features_df.shape:', train_features_df.shape)

print('train_targets_df.shape:', train_targets_df.shape)

print('test_features_df.shape:', test_features_df.shape)
train_features_df.sample(5)
train_targets_df.sample(5)
sample_submission_df = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

sample_submission_df.sample(5)
for target_name in list(train_targets_df)[1:]:

    rate = float(sum(train_targets_df[target_name])) / len(train_targets_df)

    print('%.4f percent positivity rate for %s' % (100 * rate, target_name))
num_train_samples = int(0.8 * len(train_features_df))



full_train_features_ids = train_features_df.pop('sig_id')

full_test_features_ids = test_features_df.pop('sig_id')

train_targets_df.pop('sig_id')



full_train_features_df = train_features_df.copy()

full_train_targets_df = train_targets_df.copy()



val_features_df = train_features_df[num_train_samples:]

train_features_df = train_features_df[:num_train_samples]

val_targets_df = train_targets_df[num_train_samples:]

train_targets_df = train_targets_df[:num_train_samples]



print('Total training samples:', len(full_train_features_df))

print('Training split samples:', len(train_features_df))

print('Validation split samples:', len(val_features_df))
predictions = []

for target_name in list(train_targets_df):

    rate = float(sum(train_targets_df[target_name])) / len(train_targets_df)

    predictions.append(rate)

predictions = np.array([predictions] * len(val_features_df))



targets = np.array(val_targets_df)

score = keras.losses.BinaryCrossentropy()(targets, predictions)

print('Baseline score: %.4f' % score.numpy())
feature_names = list(train_features_df)

categorical_feature_names = ['cp_type', 'cp_dose']

numerical_feature_names = [name for name in feature_names if name not in categorical_feature_names]



def merge_numerical_features(feature_dict):

    categorical_features = {name: feature_dict[name] for name in categorical_feature_names}

    numerical_features = tf.stack([tf.cast(feature_dict[name], 'float32') for name in numerical_feature_names])

    feature_dict = categorical_features

    feature_dict.update({'numerical_features': numerical_features})

    return feature_dict



train_features_ds = tf.data.Dataset.from_tensor_slices(dict(train_features_df))

train_features_ds = train_features_ds.map(lambda x: merge_numerical_features(x))

train_targets_ds = tf.data.Dataset.from_tensor_slices(np.array(train_targets_df))

train_ds = tf.data.Dataset.zip((train_features_ds, train_targets_ds))



full_train_features_ds = tf.data.Dataset.from_tensor_slices(dict(full_train_features_df))

full_train_features_ds = full_train_features_ds.map(lambda x: merge_numerical_features(x))

full_train_targets_ds = tf.data.Dataset.from_tensor_slices(np.array(full_train_targets_df))

full_train_ds = tf.data.Dataset.zip((full_train_features_ds, full_train_targets_ds))



val_features_ds = tf.data.Dataset.from_tensor_slices(dict(val_features_df))

val_features_ds = val_features_ds.map(lambda x: merge_numerical_features(x))

val_targets_ds = tf.data.Dataset.from_tensor_slices(np.array(val_targets_df))

val_ds = tf.data.Dataset.zip((val_features_ds, val_targets_ds))



test_ds = tf.data.Dataset.from_tensor_slices(dict(test_features_df))

test_ds = test_ds.map(lambda x: merge_numerical_features(x))



print('Training split samples:', int(train_ds.cardinality()))

print('Validation split samples:', int(val_ds.cardinality()))

print('Test samples:', int(test_ds.cardinality()))



train_ds = train_ds.shuffle(1024).batch(64).prefetch(8)

full_train_ds = full_train_ds.shuffle(1024).batch(64).prefetch(8)

val_ds = val_ds.batch(64).prefetch(8)

test_ds = test_ds.batch(64).prefetch(8)
from tensorflow.keras.layers.experimental.preprocessing import Normalization

from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding

from tensorflow.keras.layers.experimental.preprocessing import StringLookup





def encode_numerical_feature(feature, name, dataset):

    # Create a Normalization layer for our feature

    normalizer = Normalization()



    # Prepare a Dataset that only yields our feature

    feature_ds = dataset.map(lambda x, y: x[name])



    # Learn the statistics of the data

    normalizer.adapt(feature_ds)



    # Normalize the input feature

    encoded_feature = normalizer(feature)

    return encoded_feature





def encode_categorical_feature(feature, name, dataset):

    # Create a Lookup layer which will turn strings into integer indices

    index = StringLookup()



    # Prepare a Dataset that only yields our feature

    feature_ds = dataset.map(lambda x, y: x[name])



    # Learn the set of possible feature values and assign them a fixed integer index

    index.adapt(feature_ds)



    # Turn the values into integer indices

    encoded_feature = index(feature)



    # Create a CategoryEncoding for our integer indices

    encoder = CategoryEncoding(output_mode="binary")



    # Prepare a dataset of indices

    feature_ds = feature_ds.map(index)



    # Learn the space of possible indices

    encoder.adapt(feature_ds)



    # Apply one-hot encoding to our indices

    encoded_feature = encoder(encoded_feature)

    return encoded_feature
all_inputs = []

all_encoded_features = []



print('Processing categorical features...')

for name in categorical_feature_names:

    inputs = keras.Input(shape=(1,), name=name, dtype='string')

    encoded = encode_categorical_feature(inputs, name, train_ds)

    all_inputs.append(inputs)

    all_encoded_features.append(encoded)



print('Processing numerical features...')

numerical_inputs = keras.Input(shape=(len(numerical_feature_names),), name='numerical_features')

encoded_numerical_features = encode_numerical_feature(numerical_inputs, 'numerical_features', train_ds)



all_inputs.append(numerical_inputs)

all_encoded_features.append(encoded_numerical_features)

features = layers.Concatenate()(all_encoded_features)
x = layers.Dropout(0.5)(features)

outputs = layers.Dense(206, activation='sigmoid')(x)

basic_model = keras.Model(all_inputs, outputs)

basic_model.summary()

basic_model.compile(optimizer=keras.optimizers.RMSprop(),

                    loss=keras.losses.BinaryCrossentropy())

basic_model.fit(full_train_ds, epochs=10, validation_data=val_ds)
import kerastuner as kt



def make_model(hp):

    x = features

    num_dense = hp.Int('num_dense', min_value=0, max_value=3, step=1)

    for i in range(num_dense):

        units = hp.Int('units_{i}'.format(i=i), min_value=32, max_value=256, step=32)

        dp = hp.Float('dp_{i}'.format(i=i), min_value=0., max_value=0.5)

        x = layers.Dropout(dp)(x)

        x = layers.Dense(units, activation='relu')(x)

    

    dp = hp.Float('final_dp', min_value=0., max_value=0.5)

    x = layers.Dropout(dp)(x)

    outputs = layers.Dense(206, activation='sigmoid')(x)

    model = keras.Model(all_inputs, outputs)



    learning_rate = hp.Float('learning_rate', min_value=3e-4, max_value=3e-3)

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(loss=keras.losses.BinaryCrossentropy(),

                  optimizer=optimizer)

    model.summary()

    return model





tuner = kt.tuners.BayesianOptimization(

    make_model,

    objective='val_loss',

    max_trials=5,  # Set to 5 to run quicker, but need 100+ for good results

    overwrite=True)



callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)]

tuner.search(train_ds, validation_data=val_ds, callbacks=callbacks, epochs=100)
def get_trained_model(hp):

    model = make_model(hp)

    # First, find the best number of epochs to train for

    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=4)]

    history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)

    val_loss_per_epoch = history.history['val_loss']

    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1

    print('Best epoch: %d' % (best_epoch,))

    model = make_model(hp)

    # Increase epochs by 20% when training on the full dataset

    model.fit(full_train_ds, epochs=int(best_epoch * 1.2))

    return model
n = 2  # E.g. n=10 for top ten models

best_hps = tuner.get_best_hyperparameters(n)



all_preds = []

for hp in best_hps:

    model = get_trained_model(hp)

    preds = model.predict(test_ds)

    all_preds.append(preds)
preds = np.zeros(shape=(len(test_features_df), 206))

for p in all_preds:

    preds += p

preds /= len(all_preds)
submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



columns = list(submission.columns)

columns.remove('sig_id')



for i in range(len(columns)):

    submission[columns[i]] = preds[:, i]



submission.to_csv('submission.csv', index=False)
submission.head()