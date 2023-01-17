import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

from keras.losses import BinaryCrossentropy

import matplotlib.pyplot as plt
# df stands for dataframe

df = pd.read_csv('../input/housepricedata.csv')
# print the first 10 rows of dataset

df.head(10)
dataset = df.values
# let's print the first 5 rows of data

print(dataset[:5,:])
df.columns
# first 10 columns (not counting the name column) as features

X = dataset[:,0:10]

# last column as output

Y = dataset[:,-1]
# initiate a scaler by calling it from sklearn. Just like initiate a LinearRegressor or LogisticRegressor.

min_max_scaler = preprocessing.MinMaxScaler()



# apply scaling to our data by calling fit_transform()

X_scale = min_max_scaler.fit_transform(X)
# let's print some rows of the features's scaled version

X_scale[:5,:5]
# split 8 parts out of 10 for training

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.2)

# split 1 part out of 10 for each validation and test set

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
# training size

print('Training set:', X_train.shape, Y_train.shape)

# validation size

print('Validation set:', X_val.shape, Y_val.shape)

# test size

print('Test set:', X_test.shape, Y_test.shape)
model = Sequential([

    Dense(32, activation='relu', input_shape=(10,)),

    Dense(32, activation='relu'),

    

    Dense(1, activation='sigmoid'),

])
model.summary()
model.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True),

              loss=BinaryCrossentropy(),

              metrics=['accuracy'])
hist = model.fit(X_train, Y_train,

          batch_size=32, epochs=50,

          validation_data=(X_val, Y_val))
loss_value, metric_value = model.evaluate(X_test, Y_test)



print('Loss:', loss_value)

print('Accuracy:', metric_value)
plt.plot(hist.history['loss'], label='Train set loss values')

plt.plot(hist.history['val_loss'], label='Validation set loss values')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(loc='upper right')

plt.show()
plt.plot(hist.history['accuracy'], label='Train set accuracy')

plt.plot(hist.history['val_accuracy'], label='Validation set accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(loc='lower right')

plt.show()
model_2 = Sequential([

    Dense(1000, activation='relu', input_shape=(10,)),

    Dense(1000, activation='relu'),

    Dense(1000, activation='relu'),

    Dense(1000, activation='relu'),

    Dense(1, activation='sigmoid'),

])

model_2.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

hist_2 = model_2.fit(X_train, Y_train,

          batch_size=32, epochs=100,

          validation_data=(X_val, Y_val))
plt.plot(hist_2.history['loss'], label='Train set loss values')

plt.plot(hist_2.history['val_loss'], label='Validation set loss values')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(loc='upper right')

plt.show()
plt.plot(hist_2.history['accuracy'], label='Train set accuracy')

plt.plot(hist_2.history['val_accuracy'], label='Validation set accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(loc='lower right')

plt.show()
from keras.layers import Dropout

from keras import regularizers
model_3 = Sequential([

    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(10,)),

    Dropout(0.3),

    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),

    Dropout(0.3),

    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),

    Dropout(0.3),

    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),

    Dropout(0.3),

    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),

])

model_3.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

hist_3 = model_3.fit(X_train, Y_train,

          batch_size=32, epochs=100,

          validation_data=(X_val, Y_val))
plt.plot(hist_3.history['loss'], label='Train set loss values')

plt.plot(hist_3.history['val_loss'], label='Validation set loss values')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(loc='upper right')

plt.show()