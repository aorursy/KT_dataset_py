import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")

sns.set_style('whitegrid')

%matplotlib inline



# Load the data

df = pd.read_csv('/kaggle/input/csgo-round-winner-classification/csgo_round_snapshots.csv')



# Split X and y

y = df.round_winner

X = df.drop(['round_winner'], axis=1)



# Drop columns with grenade info

cols_grenade = 'grenade'

X = X.drop(X.columns[X.columns.str.contains(cols_grenade)], axis=1)



print(f"Total number of samples: {len(X)}")



X.head()
# Print a random snapshot as a sample

sample_index = 25

print(df.iloc[sample_index])
plt.figure(figsize=(8,6))

ax = sns.countplot(x="map", hue="round_winner", data=df)

ax.set(title='Round winners on each map')

plt.show()
plt.figure(figsize=(8,6))

ax = sns.countplot(x="map", hue="bomb_planted", data=df)

ax.set(title='Maps and bomb planted')

plt.show()
plt.figure(figsize=(8,6))

ax = sns.barplot(x=df['round_winner'].unique(), y=df['round_winner'].value_counts())

ax.set(title='Total wins per side', xlabel='Side', ylabel='Wins')

plt.show()
# Plot the distribution of health

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(12,5))

sns.distplot(df['ct_health'], bins=10, ax=ax1);

sns.distplot(df['t_health'], bins=10, ax=ax2);
# Plot the distribution of money

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(12,5))

sns.distplot(df['ct_money'], bins=10, ax=ax1);

sns.distplot(df['t_money'], bins=10, ax=ax2);
# Plot the distribution of scores

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(12,5))

sns.kdeplot(df['ct_score'], shade=True, ax=ax1)

sns.kdeplot(df['t_score'], shade=True, ax=ax2)
# Plot the distribution of time left

plt.figure(figsize=(8,6))

sns.kdeplot(df['time_left'], shade=True)
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import power_transform



def encode_targets(y):

    encoder = LabelEncoder()

    encoder.fit(y)

    y_encoded = encoder.transform(y)

    return y_encoded



def encode_inputs(X, object_cols):

    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    X_encoded = pd.DataFrame(ohe.fit_transform(X[object_cols]))

    X_encoded.columns = ohe.get_feature_names(object_cols)

    X_encoded.index = X.index

    return X_encoded



def yeo_johnson(series):

    arr = np.array(series).reshape(-1, 1)

    return power_transform(arr, method='yeo-johnson')



# Use OH encoder to encode predictors

object_cols = ['map', 'bomb_planted']

X_encoded = encode_inputs(X, object_cols)

numerical_X = X.drop(object_cols, axis=1)

X = pd.concat([numerical_X, X_encoded], axis=1)



# Use label encoder to encode targets

y = encode_targets(y)



# Make data more Gaussian-like

cols = ['time_left', 'ct_money', 't_money', 'ct_health',

 't_health', 'ct_armor', 't_armor', 'ct_helmets', 't_helmets',

  'ct_defuse_kits', 'ct_players_alive', 't_players_alive']

for col in cols:

    X[col] = yeo_johnson(X[col])
from sklearn.model_selection import train_test_split

from tensorflow import keras



# Make a train, validation and test set

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y,

 stratify=y, test_size=0.1, random_state=0)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full,

 stratify=y_train_full, test_size=0.25, random_state=0)



# Set model parameters

n_layers = 4

n_nodes = 300

regularized = False

dropout = True

epochs = 50



# Make a Keras DNN model

model = keras.models.Sequential()

model.add(keras.layers.BatchNormalization())

for n in range(n_layers):

    if regularized:

        model.add(keras.layers.Dense(n_nodes, kernel_initializer="he_normal",

         kernel_regularizer=keras.regularizers.l1(0.01), use_bias=False))

    else:

        model.add(keras.layers.Dense(n_nodes,

         kernel_initializer="he_normal", use_bias=False))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Activation("elu"))

    if dropout:

        model.add(keras.layers.Dropout(rate=0.2))

model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])



# Make a callback that reduces LR on plateau

reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                                                 patience=5, min_lr=0.001)



# Make a callback for early stopping

early_stopping_cb = keras.callbacks.EarlyStopping(patience=5)



# Train DNN.

history = model.fit(np.array(X_train), np.array(y_train), epochs=epochs,

     validation_data=(np.array(X_valid), np.array(y_valid)),

      callbacks=[reduce_lr_cb, early_stopping_cb], batch_size=128)
model.summary()
# Evaluate the test set

model.evaluate(X_test, y_test)
# Plot the loss curves for training and validation.

history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values)+1)



plt.figure(figsize=(8,6))

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# Plot the accuracy curves for training and validation.

acc_values = history_dict['accuracy']

val_acc_values = history_dict['val_accuracy']

epochs = range(1, len(acc_values)+1)



plt.figure(figsize=(8,6))

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# Predict the winning teams for ten rounds.

X_new = X_test[:10]

y_pred = model.predict_classes(X_new)

class_names = ['CT', 'T']

np.array(class_names)[y_pred]
# Show the predicated probabilities. Below 0.5 predicts CT, otherwise T.

y_proba = model.predict(X_new)

y_proba.round(2)