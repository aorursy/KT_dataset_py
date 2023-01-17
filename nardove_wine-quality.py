import numpy as np
import pandas as pd
import random

wine_df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
# wine_df = pd.read_csv('winequality-red.csv')
wine_df.describe()
wine_df.head()
field_names = list(wine_df.columns)
# field_names
X = wine_df.loc[:, ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'free sulfur dioxide', 'pH', 'alcohol']].to_numpy()
X.shape
X[0:1]
y = wine_df.loc[:, wine_df.columns == 'quality'].to_numpy()
# y -= 3
y[0:5]
mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std
num_val_samples = 750
X_train = np.copy(X[num_val_samples:num_val_samples * 2])
y_train = np.copy(y[num_val_samples:num_val_samples * 2])
print(X_train.shape, y_train.shape)
X_val = np.copy(X[:num_val_samples])
y_val = np.copy(y[:num_val_samples])
print(X_val[0], y_val[0])
X_test = np.copy(X[num_val_samples * 2:])
y_test = np.copy(y[num_val_samples * 2:])
print(X_test.shape, y_test.shape)
from keras.utils.np_utils import to_categorical

one_hot_y_train = to_categorical(y_train)
one_hot_y_val = to_categorical(y_val)
one_hot_y_test = to_categorical(y_test)

print(one_hot_y_train.shape, one_hot_y_val.shape, one_hot_y_test.shape, sep= ' | ')
one_hot_y_train[0:1]
one_hot_y_test[0:1]
one_hot_y_train[0:1]
y[0:1]
from keras import models
from keras import layers

def build_model(_units=64, _activation='relu', _activation_2='softmax', _optimizer='rmsprop', _loss='categorical_crossentropy', _metrics='accuracy'):
    model = models.Sequential()
    model.add(layers.Dense(_units, activation=_activation, input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(_units, activation=_activation))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(one_hot_y_train.shape[1], activation=_activation_2))
    model.compile(optimizer=_optimizer, loss=_loss, metrics=[_metrics])
    return model
# model = build_model()
model = build_model(128, 'sigmoid', 'softmax', 'adam')
batch_size = 16
history = model.fit(X_train, one_hot_y_train, 
          validation_data=(X_val, one_hot_y_val), 
          epochs=100,
          batch_size=batch_size,
          verbose=2)

test_loss, test_acc = model.evaluate(X_test, one_hot_y_test, batch_size=batch_size, verbose=2)
print(f'test loss: {test_loss} - test acc: {test_acc}')
for n in range(5):
    t = random.randint(1, 98) #test value
    predictions = model.predict(X_test[t:t+1])
    a = np.argmax(np.around(predictions, 1)[0])
    b = np.argmax(one_hot_y_test[t:t+1][0])
    result = 'Pass' if a == b else 'Fail'
    print(f'Test value: {t} -> {a} {b} >>> {result}\n')
history_dict = history.history
history_dict.keys()
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy', color='coral')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy', color='navy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss', color='coral')
plt.plot(epochs, val_loss, 'b', label='Validation loss', color='navy')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
