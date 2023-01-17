import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from matplotlib import pyplot as plt

import seaborn as sns



sns.set()
dftrain = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

dftest = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
dftrain.head()
dftest.head()
dftrain['pixel256'].describe()
_, ax = plt.subplots(5, 5, figsize=(8, 8))

ax = ax.flatten()

for a, (_, pixels), label in zip(ax, dftrain.filter(regex='pixel.*').iterrows(), dftrain['label']):

    a.imshow(pixels.values.reshape((28,28)), cmap="gray_r")

    a.axis('off')

    a.set_title(label)
print(f'Train size: {dftrain.shape[0]}')

print(f'Test size:  {dftest.shape[0]}')

print(f'NaNs in train: {dftrain.isna().sum().any()}')

print(f'NaNs in test:  {dftest.isna().sum().any()}')
plt.figure(figsize=(10, 10))

sns.countplot(data=dftrain, x='label');
X_train = dftrain.drop(columns='label').values

y_train = keras.utils.to_categorical(dftrain['label'].values)

X_test = dftest.values
lambda_ = 1e-2

model = keras.Sequential([

    keras.Input(shape=(X_train.shape[1],)),

    keras.layers.Dense(20, activation='relu', name='l1', kernel_regularizer=keras.regularizers.l2(lambda_)),

    keras.layers.Dense(15, activation='relu', name='l2', kernel_regularizer=keras.regularizers.l2(lambda_)),

    keras.layers.Dense(10, activation='softmax', name='output', kernel_regularizer=keras.regularizers.l2(lambda_))

])
model.compile(

    optimizer=keras.optimizers.Adam(),

    loss='categorical_crossentropy',

    metrics=['accuracy']

)
history = model.fit(

    X_train, 

    y_train, 

    validation_split=0.2, 

    batch_size=128, 

    epochs=50

)
loss = history.history['loss']

val_loss = history.history['val_loss']

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']
firstidx = 1

_, ax = plt.subplots(2, 1, figsize=(15, 15))

plt.sca(ax[0])

plt.plot(loss[firstidx:], label='loss')

plt.plot(val_loss[firstidx:], label='val_loss')

plt.legend()

plt.sca(ax[1])

plt.plot(acc[firstidx:], label='accuracy')

plt.plot(val_acc[firstidx:], label='val_accuracy')

plt.legend()
dftrain_pred = dftrain.copy()

dftrain_pred['pred'] = model.predict(X_train).argmax(axis=1)
success_example = dftrain_pred[dftrain_pred['label'] == dftrain_pred['pred']].iloc[0]

err_example = dftrain_pred[dftrain_pred['label'] != dftrain_pred['pred']].iloc[0]



_, ax = plt.subplots(1, 2, figsize=(6, 6))

ax = ax.flatten()



for a, example in zip(ax, (success_example, err_example)):

    a.imshow(example.filter(regex='pixel*').values.reshape((28,28)), cmap="gray_r")

    a.axis('off')

    a.set_title(f'Predicted: {example["pred"]}, actual: {example["label"]}')
preds_prob = model.predict(X_test)

preds_label = preds_prob.argmax(axis=1)
dfpreds = pd.DataFrame({

    'ImageId': np.arange(len(preds_label)) + 1,

    'Label': preds_label

})

dfpreds.to_csv('submissions.csv', index=False)