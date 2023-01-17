import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 100)

import tensorflow as tf

import random



seed = 52

tf.random.set_seed(seed)

random.seed(seed)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_df = pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')
data_df.sample(5)
data_df.histogram_tendency.value_counts()
data_df = pd.get_dummies(data_df, prefix='ht', columns=['histogram_tendency'])
data_df.info()
print(data_df.fetal_health.value_counts())

data_df.fetal_health.value_counts().plot(kind='pie')
y = data_df.fetal_health.values.astype(int)

X = data_df.drop('fetal_health', axis='columns').values
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(y)

y = tf.keras.utils.to_categorical(y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
tf.keras.backend.clear_session()

from tensorflow.keras import Input, Model

from tensorflow.keras.layers import Dense, BatchNormalization, GaussianNoise



inp = Input(shape=(x_train.shape[1]))

x = BatchNormalization()(inp)

x = Dense(512, activation='relu')(x)

x = BatchNormalization()(x)

x = Dense(512, activation='relu')(x)

x = BatchNormalization()(x)

x = Dense(512, activation='relu')(x)

x = BatchNormalization()(x)

x = Dense(512, activation='relu')(x)

x = BatchNormalization()(x)

out = Dense(3, activation='softmax')(x)



model = Model(inputs=[inp], outputs=[out])

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])



es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1)



history = model.fit(x_train, y_train, epochs=200, callbacks=[es], validation_split=0.2)
import matplotlib.pyplot as plt

import seaborn as sns



history = history.history



fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))



ax1.plot(history['loss'], label='Training')

ax1.plot(history['val_loss'], label='Validation')

ax1.legend(loc='best')

ax1.set_title('Loss')



ax2.plot(history['acc'], label='Training')

ax2.plot(history['val_acc'], label='Validation')

ax2.legend(loc='best')

ax2.set_title('Accuracy')



plt.xlabel('Epochs')

sns.despine()

plt.show()
model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
y_true = y_test
m = tf.keras.metrics.AUC(curve='ROC')

m.update_state(y_true, y_pred)

print('AUROC = ' + str(m.result().numpy()))
import tensorflow_addons as tfa
m = tfa.metrics.F1Score(num_classes=3)

m.update_state(y_true, y_pred)

f1 = m.result().numpy()

print('F1 Scores')

print('')

print('Normal: ' + str(f1[0]))

print('Suspect: ' + str(f1[1]))

print('Pathological: ' + str(f1[2]))

m = tf.keras.metrics.AUC(curve='PR')

m.update_state(y_true, y_pred)

print('AUPRC = ' + str(m.result().numpy()))