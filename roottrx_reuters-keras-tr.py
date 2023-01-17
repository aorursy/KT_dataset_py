# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000) 
# num_words = 10000 parametresi verilen en sık karşılaşılan 10000 kelime ile sınırlı olmasını sağlar.
len(train_data)
len(test_data)
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

# eğitim verisinin vektöre dönüştürülmesi 
X_train = vectorize_sequences(train_data)

# Test verisinin vektöre dönüştürülmesi
X_test = vectorize_sequences(test_data)
# Eğitim etiketlerinin vektöre dönüştürülmesi
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
# Model Tanımı
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10000,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))
# Modeli derlemek 
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Doğrulama veri seti oluşturma 
X_val = X_train[:1000]
partial_X_train = X_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
# Model Eğitmek
history = model.fit(partial_X_train, 
                    partial_y_train,
                    epochs=20, 
                    batch_size=512,
                    validation_data=(X_val, y_val))
# Eğitim ve Doğrulama kayıplarını çizdirmek

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Eğitim Kaybı')
plt.plot(epochs, val_loss, 'b', label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epoklar')
plt.ylabel('Kayıp')
plt.legend()

plt.show()

plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epok')
plt.ylabel('Başarım')
plt.legend()
plt.show()
# Modeli en baştan eğitmek
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10000,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_X_train, 
          partial_y_train, 
          epochs=9,
          batch_size=512,
          validation_data=(X_val, y_val))

results = model.evaluate(X_test, one_hot_test_labels)
results
predictions = model.predict(X_test)