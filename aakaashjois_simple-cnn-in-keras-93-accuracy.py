import os
import keras
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.preprocessing import normalize

py.init_notebook_mode(connected=True)
train = pd.read_csv('../input/fashion-mnist_train.csv')
test = pd.read_csv('../input/fashion-mnist_test.csv')
train_labels = train.label
train_data = train.drop(columns=['label'])
test_labels = test.label
test_data = test.drop(columns=['label'])
train_data = normalize(train_data)
test_data = normalize(test_data)
train_data = train_data.reshape((-1, 28, 28, 1))
test_data = test_data.reshape((-1, 28, 28, 1))
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same', input_shape = (28, 28, 1), bias_initializer=keras.initializers.Constant(1e-3), kernel_regularizer='l2'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Conv2D(128, 3, activation='relu', padding='same', bias_initializer=keras.initializers.Constant(1e-3), kernel_regularizer='l2'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Conv2D(256, 3, activation='relu', padding='same', bias_initializer=keras.initializers.Constant(1e-3), kernel_regularizer='l2'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Conv2D(512, 3, activation='relu', padding='same', bias_initializer=keras.initializers.Constant(1e-3), kernel_regularizer='l2'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.GlobalMaxPooling2D())
model.add(keras.layers.Dense(512, activation='relu', bias_initializer=keras.initializers.Constant(1e-3), kernel_regularizer='l2'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x=train_data,
                    y=train_labels,
                    validation_split=0.2,
                    batch_size=512,
                    epochs=50,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                             mode='auto',
                                                             patience=10,
                                                             verbose=1), 
                               keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                 patience=3,
                                                                 min_lr=1e-5, 
                                                                 factor=0.2,
                                                                 verbose=1)],
                    shuffle=True,
                    verbose=2)
loss, acc = model.evaluate(x=test_data, y=test_labels, batch_size=512)
print('Test loss: {} - Test accuracy - {}'.format(loss, acc))
py.iplot(dict(data=[
            go.Scatter(y=history.history['loss'], name='Training loss'),
            go.Scatter(y=history.history['acc'], name='Training accuracy'),
            go.Scatter(y=history.history['val_loss'], name='Validation loss'),
            go.Scatter(y=history.history['val_acc'], name='Validation accuracy')],
         layout=dict(title='Training and validation history', 
                     xaxis=dict(title='Epochs'), 
                     yaxis=dict(title='Value'))),
         filename='plot')