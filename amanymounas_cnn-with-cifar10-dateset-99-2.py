import tensorflow as tf
import os
import numpy as np
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt
%matplotlib inline

if not os.path.isdir('models'):
    os.mkdir('models')
    

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())
print('TensorFlow version:', tf.__version__)

def get_three_classes(x, y):
    indices_0, _ = np.where(y == 0.)
    indices_1, _ = np.where(y == 1.)
    indices_2, _ = np.where(y == 2.)

    indices = np.concatenate([indices_0, indices_1, indices_2], axis=0)
    
    x = x[indices]
    y = y[indices]
    
    count = x.shape[0]
    #randome and unique
    indices = np.random.choice(range(count), count, replace=False)
    
    x = x[indices]
    y = y[indices]
    
    y = tf.keras.utils.to_categorical(y)
    
    return x, y
(x_train, y_train), (x_test, y_test)= tf.keras.datasets.cifar10.load_data()

x_train,y_train= get_three_classes(x_train,y_train)
x_test, y_test= get_three_classes(x_test, y_test)

print(x_train.shape,y_train.shape)
print(x_test.shape, y_test.shape)
class_names = ['aeroplane', 'car', 'bird']
def show_randome_example(x,y,p):
    indices = np.random.choice(range(x.shape[0]), 10, replace=False)
    
    x = x[indices]
    y = y[indices]
    p = p[indices]
    
    plt.figure(figsize=(10,5))
    for i in range(10):
            plt.subplot(2,5,1+i)
            plt.imshow(x[i])
            plt.xticks([])
            plt.yticks([])
            col = 'green' if np.argmax(y[i]) == np.argmax(p[i]) else 'red'
            plt.xlabel(class_names[np.argmax(p[i])], color=col)
    plt.show()
        
show_randome_example(x_train, y_train, y_train)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense


def create_model():

    def add_conv_block(model, num_filters, input_shape=None):

        if input_shape:
            model.add(Conv2D(num_filters, 3, activation='relu', padding='same', input_shape=input_shape))
        else:
            model.add(Conv2D(num_filters, 3, activation='relu', padding='same'))

        model.add(BatchNormalization())
        model.add(Conv2D(num_filters, 3, activation='relu', padding='valid'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        return model

    model = tf.keras.models.Sequential()
    model = add_conv_block(model, 32, input_shape=(32, 32, 3))
    model = add_conv_block(model, 64)
    model = add_conv_block(model, 128)

    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = create_model()
model.summary()
%%time

h = model.fit(
    x_train/255., y_train,
    validation_data=(x_test/255., y_test),
    epochs=10, batch_size=128,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2),
        tf.keras.callbacks.ModelCheckpoint('models/model_{val_accuracy:.3f}.h5', save_best_only=True,
                                          save_weights_only=False, monitor='val_accuracy')
    ]
)

losses = h.history['loss']
accs = h.history['accuracy']
val_losses = h.history['val_loss']
val_accs = h.history['val_accuracy']
epochs = len(losses)

plt.figure(figsize=(12, 4))
for i, metrics in enumerate(zip([losses, accs], [val_losses, val_accs], ['Loss', 'Accuracy'])):
    plt.subplot(1, 2, i + 1)
    plt.plot(range(epochs), metrics[0], label='Training {}'.format(metrics[2]))
    plt.plot(range(epochs), metrics[1], label='Validation {}'.format(metrics[2]))
    plt.legend()
plt.show()
model = tf.keras.models.load_model('./models/model_0.920.h5')
preds = model.predict(x_test/255.)
show_randome_example(x_test, y_test, preds)