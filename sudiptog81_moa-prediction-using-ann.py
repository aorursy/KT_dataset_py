import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from colorama import Fore, Style

sns.set()
import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss, classification_report, confusion_matrix
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
train_features.head().T
train_targets_scored.head().T
train_targets_nonscored.head().T
test_features.head().T
test_ids = test_features['sig_id']
for d in [train_features, test_features]:

    d.drop(['sig_id', 'cp_type', 'cp_dose', 'cp_time'], axis=1, inplace=True)

    

train_features.head().T
train_targets_scored.drop(['sig_id'], axis=1, inplace=True)

train_targets_scored.head().T
fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for i, column in enumerate(train_features[train_features.columns[:9]].columns):

    sns.distplot(train_features[column], ax=axes[i // 3, i % 3])

plt.tight_layout()

plt.show()
fig, axes = plt.subplots(3, 3, figsize=(8, 8))

for i, column in enumerate(train_features[train_features.columns[800:809]].columns):

    sns.distplot(train_features[column], ax=axes[i // 3, i % 3], color='green')

plt.tight_layout()

plt.show()
print(Fore.YELLOW + 'Shape(x_train): ' + str(train_features.shape))

print(Fore.YELLOW + 'Shape(y_train): ' + str(train_targets_scored.shape))

print(Fore.BLUE + 'Shape(x_test): ' + str(test_features.shape))
x_train, x_cv, y_train, y_cv = train_test_split(train_features, train_targets_scored, test_size=0.2) 
model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(512, input_dim=x_train.shape[1], activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Dense(1024, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(.25),

    tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')

])



model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)



model.summary()
history = model.fit(

    x_train, y_train, verbose=2, epochs=40,

    validation_data=(x_cv, y_cv),

    callbacks=[

        tf.keras.callbacks.ReduceLROnPlateau(

            monitor='val_loss', 

            factor=0.2, 

            patience=4

        ),

        tf.keras.callbacks.EarlyStopping(

            monitor='val_loss',

            patience=10,

            mode='auto',

            verbose=1,

            baseline=None,

            restore_best_weights=True

        )

    ]

)
loss_train = history.history['loss']

loss_validation = history.history['val_loss']

epochs = range(1, len(history.history['loss']) + 1)

plt.plot(epochs, loss_train, 'g', label='Training')

plt.plot(epochs, loss_validation, 'b', label='Validation')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Loss')

plt.legend()

plt.show()
acc_train = history.history['accuracy']

acc_validation = history.history['val_accuracy']

epochs = range(1, len(history.history['accuracy']) + 1)

plt.plot(epochs, acc_train, 'g', label='Training')

plt.plot(epochs, acc_validation, 'b', label='Validation')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Accuracy')

plt.legend()

plt.show()
print(Fore.BLUE + f'Average LogLoss: {log_loss(y_cv, model.predict(x_cv)) / 207}')
y_hat = model.predict(test_features)
submission = pd.concat(

    [pd.DataFrame(test_ids, columns=['sig_id']),

     pd.DataFrame(y_hat, columns=train_targets_scored.columns)],

    axis=1

)

submission.head().T
submission.to_csv('submission.csv',index=False)