import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv')
data.head()
def remove_characters(value):
    new_value = '0'
    value = str(value)[2:]
    for i in value:
        if i.isdigit():
            new_value += i
    return int(new_value)
def change_floor(value):
    if value == '-':
        value = 0
    return value
data['hoa'] = pd.to_numeric(data['hoa'].apply(remove_characters), errors= 'ignore')
data['rent amount'] = pd.to_numeric(data['rent amount'].apply(remove_characters), errors= 'ignore')
data['property tax'] = pd.to_numeric(data['property tax'].apply(remove_characters), errors= 'ignore')
data['fire insurance'] = pd.to_numeric(data['fire insurance'].apply(remove_characters), errors= 'ignore')
data['total'] = pd.to_numeric(data['total'].apply(remove_characters), errors= 'ignore')
data['floor'] = pd.to_numeric(data['floor'].apply(change_floor), errors= 'ignore')
data.head()
data['animal'] = data['animal'].map({'acept': 1, 'not acept': 0})
data['furniture'] = data['furniture'].map({'furnished': 1, 'not furnished': 0})
data.head()
print(data['rooms'].value_counts())
print(data['bathroom'].value_counts())
labels_dict = {
    'city': [1, 0],
    'animal': ['Yes', 'No'],
    'furniture': ['Furnished', 'Not furnished'],
    'rooms': [str(i) for i in range(3, 0, -1)] + [str(i) for i in range(4, 9)] + ['10'],
    'bathroom': [str(i) for i in range(1, 8)] + ['9', '8', '10']
}
count = 1
for key in list(labels_dict.keys())[:3]:
    plt.subplot(1, 3, count)
    labels = labels_dict[key]
    values = data[key].value_counts()
    colors = ['blue', 'cyan']
    plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, explode=(0.1, 0.1))
    plt.title(key)
    plt.axis('equal')
    count += 1
plt.show()
count = 1
for key in list(labels_dict.keys())[3:]:
    plt.subplot(1, 2, count)
    labels = labels_dict[key]
    values = data[key].value_counts()
    plt.bar(labels, values, width=0.5, alpha=0.6, bottom=2, linewidth=2)
    plt.title(key)
    count += 1
plt.show()
y_train = np.array(data['total'])
data.drop(['total'], axis='columns', inplace=True)
data.drop(['Unnamed: 0'], axis='columns', inplace=True)
x_train = np.array(data).astype('float32')
x_train
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_train
test_split = x_train.shape[0] - int(x_train.shape[0] * 0.2)
x_test = x_train[test_split:]
y_test = y_train[test_split:]
x_train = x_train[:test_split]
y_train = y_train[:test_split]
best_model_path = 'best_model.h5'
checkpoint_callback = ModelCheckpoint(best_model_path,
                                     monitor='val_mae',
                                     save_best_only=True,
                                     verbose=1)
reduce_callback = ReduceLROnPlateau(monitor='val_mae',
                                   patience=3,
                                   factor=0.5,
                                   min_lr=0.00001,
                                   verbose=1)
callbacks_list = [checkpoint_callback, reduce_callback]
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])
history = model.fit(x_train,
                    y_train,
                    epochs=100,
                    validation_split=0.2,
                    batch_size=20,
                    callbacks=callbacks_list,
                    verbose=1)
model.load_weights(best_model_path)
testing_model = model.evaluate(x_test,
                              y_test,
                              verbose=1)
print('Mean absolute error = ' + str(int(testing_model[1] * 10) / 10) + '$')
plt.plot(history.history['mae'], 
         label='Mean absolute error')
plt.plot(history.history['val_mae'],
         label='Val mae')
plt.xlabel('Epoch')
plt.ylabel('Percentage of correct responses')
plt.legend()
plt.show()
plt.plot(history.history['loss'], 
         label='Loss')
plt.plot(history.history['val_loss'],
         label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Percentage of loss')
plt.legend()
plt.show()