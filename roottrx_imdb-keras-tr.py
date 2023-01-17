# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.datasets import imdb
from keras import models
from keras import layers

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# num_words parametresi ile en sık tekrar eden 10000 örneği saklayıp nadir örnekler göz ardı edilir.
train_data[0]
train_labels[0] # 0 = olumsuz, 1 = olumlu kriter
def vectorize_sequences(sequences, dimension=10000):
    # (len(sequences), dimension) şeklinde tüm elemanları 0 olan matris oluşturur.
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1 # result[i]'nin istenen indekslerini 1 yapar.
    return results

X_train = vectorize_sequences(train_data) # eğitim vektör verisi
X_test = vectorize_sequences(test_data) # test vektör verisi
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Elle Ayarlama: 
# from keras import losses
# from keras import metrics

# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])
# 
X_val = X_train[:10000]
partial_X_train = X_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
history = model.fit(partial_X_train, partial_y_train,
                    epochs=20, batch_size=512,
                    validation_data=(X_val, y_val))
history_dict = history.history
history_dict.keys()
# Eğitim ve doğruluk kayıplarını Çizdirmek
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Eğitim Kaybı') # bo mavi nokta için
plt.plot(epochs, val_loss_values, 'b', label='Doğruluk Kaybı')
plt.title('Eğitim ve Doğruluk Kaybı')
plt.xlabel('Epoklar')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# Eğitim ve Doğrulama başarımını çizdirmek

plt.clf() # Şekli temizler
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Eğitim Başarımı')
plt.plot(epochs, val_acc, 'b', label='Doğruluk Başarımı')
plt.title('Eğitim ve Doğruluk başarımı')
plt.xlabel("Epoklar")
plt.ylabel('Başarım')
plt.show()
# Modeli en baştan eğitmek 
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(X_test, y_test)
results
model.predict(X_test)
model = models.Sequential()
model.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))
model.add(layers.Dense(32, activation='tanh'))
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(X_test, y_test)
print(results)
