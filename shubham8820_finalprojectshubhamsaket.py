import tensorflow as tf
from matplotlib import pyplot

from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation

from sklearn.model_selection import train_test_split

import numpy as np
import random
import matplotlib.pyplot as plt

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

labels = ['zero',
          'one',
          'two',
          'three',
          'four',
          'five',
          'six',
          'seven',
          'eight',
          'nine']
print('X_train set shape of {}'.format(X_train.shape))
print('X_test set shape of {}'.format(X_test.shape))
print('y_train set shape of {}'.format(y_train.shape))
print('y_test set shape of {}'.format(y_test.shape))
img_rows, img_cols, channels = 28, 28, 1
num_classes = 10
X_train = X_train / 255
X_test = X_test / 255
X_train = X_train.reshape((-1, img_rows, img_cols, channels))
X_test = X_test.reshape((-1, img_rows, img_cols, channels))

print('X_train set shape of {}'.format(X_train.shape))
print('X_test set shape of {}'.format(X_test.shape))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print('y_train set shape of {}'.format(y_train.shape))
print('y_test set shape of {}'.format(y_test.shape))
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)

print('X_train set shape of {}'.format(X_train.shape))
print('X_val set shape of {}'.format(X_val.shape))
print('y_train set shape of {}'.format(y_train.shape))
print('y_val set shape of {}'.format(y_val.shape))
def create_model(img_rows, img_cols, channels):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu', input_shape=(img_rows, img_cols, channels)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model
model = create_model(img_rows, img_cols, channels)
model.fit(X_train,
          y_train,
         batch_size=32,
         epochs=32,
         validation_data=(X_val, y_val))
print('Base accuracy on regular images: ', model.evaluate(X_test, y_test, verbose=0))

def adversarial_pattern(image, label):
    # Step 1
    image = tf.cast(image, tf.float32)
    
    # Step 2
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        # The loss function has to be the same that the Neural Network was trained with.
        loss = tf.keras.losses.MSE(label, prediction)
    
    # Step 3
    gradient = tape.gradient(loss, image)
    
    # Step 4
    signed_grad = tf.sign(gradient)
    
    # Step 5
    return signed_grad

# A regular image from the train set will be used with its corresponding label
image = X_train[11]
image_label = y_train[11]
# Generate the adversarial perturbations
perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), image_label).numpy()
adversarial = image + perturbations * 0.1
print('The true label was: {}'.format(labels[model.predict(image.reshape((1, img_rows, img_cols, channels))).argmax()]))
print('The prediction after the attack is: {}'.format(labels[model.predict(adversarial).argmax()]))

if channels == 1:
    plt.imshow(adversarial.reshape((img_rows, img_cols)))
else:
    plt.imshow(adversarial.reshape((img_rows, img_cols, channels)))
plt.imshow(image.reshape((img_rows, img_cols)))
def generate_adversarial_samples(batchsize, X_set, y_set):
    # Step 2
    while True:
        x = []
        y = []
        # Step 3
        for batch in range(batchsize):
            N = len(y_set) - 1
            N = random.randint(0, N)
            label = y_set[N]
            image = X_set[N]
            
            # Step 4
            perturbations = adversarial_pattern(image.reshape((1, img_rows, img_cols, channels)), label).numpy()
            
            # Step 5
            epsilon = 0.1
            
            # Step 6
            adversarial = image + perturbations * epsilon
            
            # Step 7
            x.append(adversarial)
            y.append(y_set[N])
        
        # Step 8.a
        x = np.asarray(x).reshape((batchsize, img_rows, img_cols, channels))
        y = np.asarray(y)
        
        # Step 8.b
        yield x, y
X_adversarial_test, y_adversarial_test = next(generate_adversarial_samples(10000, X_test, y_test))
print('Base accuracy on adversarial images: {}'.format(model.evaluate(X_adversarial_test, y_adversarial_test, verbose=0)))
X_adversarial_train, y_adversarial_train = next(generate_adversarial_samples(54000, X_train, y_train))
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)

np.save('X_adversarial_train.npy', X_adversarial_train)
np.save('y_adversarial_train.npy', y_adversarial_train)

np.save('X_adversarial_test.npy', X_adversarial_test)
np.save('y_adversarial_test.npy', y_adversarial_test)
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

X_adversarial_train = np.load('X_adversarial_train.npy')
y_adversarial_train = np.load('y_adversarial_train.npy')

X_adversarial_test = np.load('X_adversarial_test.npy')
y_adversarial_test = np.load('y_adversarial_test.npy')

model.fit(X_adversarial_train,
          y_adversarial_train,
          batch_size=32,
          epochs=32,
          validation_data=(X_val, y_val))
print('Defended accuracy on adversarial images: {}' .format(model.evaluate(X_adversarial_test, y_adversarial_test, verbose=0)))
print('Defended accuracy on regular images: {}' .format(model.evaluate(X_test, y_test, verbose=0)))
X_train = np.row_stack((X_adversarial_train, X_train))
X_train.shape
y_train = np.row_stack((y_adversarial_train, y_train))

y_train.shape

model.fit(X_train,
          y_train,
          batch_size=32,
          epochs=32,
          validation_data=(X_val, y_val))
print('Defended accuracy on adversarial images: {}' .format(model.evaluate(X_adversarial_test, y_adversarial_test, verbose=0)))
print('Defended accuracy on regular images: {}' .format(model.evaluate(X_test, y_test, verbose=0)))
X_final_set = np.row_stack((X_adversarial_test, X_test))
y_final_set = np.row_stack((y_adversarial_test, y_test))
X_final_set.shape
y_final_set.shape
print('Defended accuracy on regular and adversarial images: {}' .format(model.evaluate(X_final_set, y_final_set, verbose=0)))