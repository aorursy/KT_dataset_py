import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
train_fname = '../input/digit-recognizer/train.csv'



df_train = pd.read_csv(train_fname)

df_train.head()
df_train.shape
df_train['label'].unique()
fig, ax = plt.subplots()



hist, bin_edges  = np.histogram(df_train['label'])



# atur posisi x-label

bin_edges = np.ceil(bin_edges)

ax.set_xticks(bin_edges[:-1])



# tambah title

ax.set_title('Histogram')

ax.set_xlabel('Digits')



# plot histogram sebagai plot bar

ax.bar(bin_edges[:-1], hist);
test_fname = '../input/digit-recognizer/test.csv'



df_test = pd.read_csv(test_fname)

df_test.head()
df_test.shape
sample_fname = '../input/digit-recognizer/sample_submission.csv'



df_sample = pd.read_csv(sample_fname)

df_sample.head()
df_sample.shape
y_train = df_train['label'].to_numpy()



del df_train['label']

X_train = df_train.to_numpy()



X_test = df_test.to_numpy()
y_train.shape, X_train.shape, X_test.shape
X_train = X_train / 255

X_test = X_test / 255
from tensorflow.keras.utils import to_categorical



y_train = to_categorical(y_train)
from tensorflow.keras import models

from tensorflow.keras import layers



network = models.Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

network.add(layers.Dense(10, activation='softmax'))
network.summary()
network.compile(optimizer='rmsprop',

                loss='categorical_crossentropy',

                metrics=['accuracy'])
network.fit(X_train, y_train, epochs=5, batch_size=128,

            validation_split=0.2);
y_test = network.predict_classes(X_test)
y_test.shape
df_pred = pd.DataFrame({

    'ImageId': range(1, y_test.shape[0] + 1),

    'Label': y_test

})
df_pred.head()
df_pred.shape
df_pred.to_csv('submission.csv', index=False)