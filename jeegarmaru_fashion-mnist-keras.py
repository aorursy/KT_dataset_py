# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

print('Keras version: %s' % keras.__version__)
print('Tensorflow version: %s' % tf.__version__)


# Any results you write to the current directory are saved as output.
width = 28
height = 28
channels = 1
classes = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
          5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot' }
num_classes = len(classes)

# Model parameters
gen_data = False
epochs = 40
batch_size = 64
def read_file(filename):
    df = pd.read_csv(f'../input/{filename}')
    y = keras.utils.to_categorical(df['label'].values, num_classes=num_classes)
    X = df.drop('label', axis=1).values
    X = X.astype('float32') / 255
    if (keras.backend.image_data_format() == 'channels_last'):
        X = X.reshape(X.shape[0], width, height, channels)
    else:
        X = X.reshape(X.shape[0], channels, width, height)
    return (X, y)
X_train, y_train = read_file('fashion-mnist_train.csv')
X_test, y_test = read_file('fashion-mnist_test.csv')
def show_images(X, locs, y, preds=None):
    if isinstance(locs, int):
        locs = [locs]
    fig=plt.figure(figsize=(15, 10))
    for i, val in enumerate(locs):
        fig.add_subplot(1, len(locs), i+1)
        plt.imshow(X[val].reshape((28, 28)))
        plt.text(0, 32, "Class: %s" % classes[np.argmax(y[val])])
        if preds is not None:
            plt.text(0, 35, "Predicted: %s" % classes[np.argmax(preds[val])])
    plt.show()
show_images(X_test, [100, 200, 300], y_test)
def create_model(conv_nodes, dense_nodes, loss='categorical_crossentropy',
                 optimizer='adam', dropout=0.2):
    model = Sequential()
    for i in range(len(conv_nodes)):
        model.add(Conv2D(conv_nodes[i], (3, 3), input_shape=(width, height, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))

    model.add(Flatten())
    
    for i in range(len(dense_nodes)):
        model.add(Dense(dense_nodes[i]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    #adam = Adam(lr=0.01, decay=1e-6)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model
model = create_model(conv_nodes=[64, 32], dense_nodes=[256, 512])
model.summary()
def execute_model(X, y, batch_size=batch_size, epochs=epochs):
    es = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=1)
    checkpoint = ModelCheckpoint('model.hdf5', verbose=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    if gen_data:
        im_gen = ImageDataGenerator(rotation_range=5, width_shift_range=0.05, height_shift_range=0.05, 
                                    brightness_range=(-0.05, 0.05), shear_range=2, zoom_range=0.05, validation_split=0.05)
        return model.fit_generator(im_gen.flow(X_train, y_train, batch_size=32), epochs=epochs, 
                                   callbacks=[es, checkpoint], validation_data=(X_val, y_val), verbose=1)
    else:
        return model.fit(X_train, y_train, batch_size=32, epochs=epochs, 
                         callbacks=[es, checkpoint], validation_data=(X_val, y_val), verbose=1)

history = execute_model(X_train, y_train)
fig, axes = plt.subplots(1, 2)
for ax, label in zip(axes, ['loss', 'acc']):
    ax.plot(history.history[label], label='Training')
    ax.plot(history.history['val_'+label], label='Validation')
    ax.legend()
    ax.set_xlabel('epochs')
    ax.set_ylabel(label)
    ax.set_title(label.upper())
fig.set_size_inches(12, 5)
score = model.evaluate(X_test, y_test, batch_size=64)
print(f"Final score is {score}")
preds = model.predict_classes(X_test)
expected = [x.argmax() for x in y_test]
results = pd.DataFrame({'preds':preds, 'expected':expected})
results['wrong_ones'] = results['preds'] != results['expected']
results.head()
wrong_results = results[results['wrong_ones']]
wrong_locs = wrong_results.index.values
def compare(locs, size=5):
    show_images(X_test, locs[np.random.randint(0, len(locs), size=size)], y_test, preds)
compare(wrong_locs)
most_wrong_class = wrong_results['expected'].value_counts().index[0]
most_wrong_results = wrong_results[wrong_results['expected'] == most_wrong_class]
most_wrong_locs = most_wrong_results.index
compare(most_wrong_locs)
model.predict_proba(X_test)