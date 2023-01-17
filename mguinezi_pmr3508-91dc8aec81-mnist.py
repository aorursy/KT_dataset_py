import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# device_name = tf.test.gpu_device_name()

# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')

# print('Found GPU at: {}'.format(device_name))
!cd ../input && ls
train_pure = np.load('../input/atividade-4-versao-2-fashion-mnist/train_images_pure.npy')
train_rotated = np.load('../input/atividade-4-versao-2-fashion-mnist/train_images_rotated.npy')
train_noisy = np.load('../input/atividade-4-versao-2-fashion-mnist/train_images_noisy.npy')
train_both = np.load('../input/atividade-4-versao-2-fashion-mnist/train_images_both.npy')
test = np.load('../input/atividade-4-versao-2-fashion-mnist/Test_images.npy')
train_pure.shape
y_train = pd.read_csv('../input/atividade-4-versao-2-fashion-mnist/train_labels.csv')
#print(y_train['label'])
i = 0
temp_y = []
for label in y_train['label']:
    temp_y.append([0]*10) 
    temp_y[-1][label] = 1
    # print(temp_y[-1])
y_train = np.array(temp_y)
y_train.shape
fig=plt.figure(figsize=(16,16))
fig.add_subplot(1, 6, 1)
imgplot = plt.imshow(train_pure[23])
fig.add_subplot(1, 6, 2)
imgplot = plt.imshow(train_rotated[23])
fig.add_subplot(1, 6, 3)
imgplot = plt.imshow(train_noisy[23])
fig.add_subplot(1, 6, 4)
imgplot = plt.imshow(train_pure[55])
fig.add_subplot(1, 6, 5)
imgplot = plt.imshow(train_rotated[55])
fig.add_subplot(1, 6, 6)
imgplot = plt.imshow(train_noisy[55])
plt.show()
fig=plt.figure(figsize=(16,16))
fig.add_subplot(1, 6, 1)
imgplot = plt.imshow(test[23])
fig.add_subplot(1, 6, 4)
imgplot = plt.imshow(test[55])
plt.show()
x_train = train_pure
x_train = np.expand_dims(x_train,axis=3)
x_train = x_train.astype('float32') / 255
x_train.shape
# modelwithoutpooling

modelwithoutpooling = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
modelwithoutpooling.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
# modelwithoutpooling.add(tf.keras.layers.MaxPooling2D(pool_size=2))
modelwithoutpooling.add(tf.keras.layers.Dropout(0.3))

modelwithoutpooling.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
#modelwithoutpooling.add(tf.keras.layers.MaxPooling2D(pool_size=2))
modelwithoutpooling.add(tf.keras.layers.Dropout(0.3))

modelwithoutpooling.add(tf.keras.layers.Flatten())
modelwithoutpooling.add(tf.keras.layers.Dense(256, activation='relu'))
modelwithoutpooling.add(tf.keras.layers.Dropout(0.5))
modelwithoutpooling.add(tf.keras.layers.Dense(10, activation='softmax'))

modelwithoutpooling.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# Take a look at the model summary
modelwithoutpooling.summary()
model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# Take a look at the model summary
model.summary()
modelwithoutpooling.fit(x_train,
         y_train,
         batch_size=64,
         epochs=5)
model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=5)
x_test = train_rotated
x_test = np.expand_dims(x_test,axis=3)
x_test = x_test.astype('float32') / 255
x_test.shape
# Evaluate the model on test set
score = modelwithoutpooling.evaluate(x_test, y_train, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])
x_test2 = train_noisy
x_test2 = np.expand_dims(x_test2,axis=3)
x_test2 = x_test2.astype('float32') / 255
x_test2.shape
# Evaluate the model on test set
score = modelwithoutpooling.evaluate(x_test2, y_train, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])
x_test3 = train_both
x_test3 = np.expand_dims(x_test3,axis=3)
x_test3 = x_test3.astype('float32') / 255
x_test3.shape
# Evaluate the model on test set
score = modelwithoutpooling.evaluate(x_test3, y_train, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])
x_test = train_rotated
x_test = np.expand_dims(x_test,axis=3)
x_test = x_test.astype('float32') / 255
x_test.shape
# Evaluate the model on test set
score = model.evaluate(x_test, y_train, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])
x_test2 = train_noisy
x_test2 = np.expand_dims(x_test2,axis=3)
x_test2 = x_test2.astype('float32') / 255
x_test2.shape
# Evaluate the model on test set
score = model.evaluate(x_test2, y_train, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])
x_test3 = train_both
x_test3 = np.expand_dims(x_test3,axis=3)
x_test3 = x_test3.astype('float32') / 255
x_test3.shape
# Evaluate the model on test set
score = model.evaluate(x_test3, y_train, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')
fig=plt.figure(figsize=(16,16))
fig.add_subplot(1, 6, 1)
imgplot = plt.imshow(train_pure[23])
fig.add_subplot(1, 6, 2)
imgplot = plt.imshow(train_rotated[23])
plt.show()
# grew_train_data = []
# for x in train_pure:
#     # x = train_pure[23]  # this is a Numpy array with shape (3, 150, 150)
#     x = x.reshape((1,) + x.shape)
#     x = x.reshape((1,) + x.shape)
#     # print(x.shape)
#     grew_train_data.append(datagen.flow(x))
# # grew_train_data[0][0][0]
# print(len(grew_train_data),len(grew_train_data[0]),len(grew_train_data[0][0]))
grew_train_data = train_pure
grew_train_data = np.expand_dims(grew_train_data,axis=3)
grew_train_data = grew_train_data.astype('float32') / 255
grew_train_data.shape
model_grew = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model_grew.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model_grew.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model_grew.add(tf.keras.layers.Dropout(0.3))

model_grew.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model_grew.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model_grew.add(tf.keras.layers.Dropout(0.3))

model_grew.add(tf.keras.layers.Flatten())
model_grew.add(tf.keras.layers.Dense(256, activation='relu'))
model_grew.add(tf.keras.layers.Dropout(0.5))
model_grew.add(tf.keras.layers.Dense(10, activation='softmax'))

model_grew.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# Take a look at the model summary
model_grew.summary()
datagen.fit(grew_train_data)
model_grew.fit_generator(datagen.flow(grew_train_data, y_train, batch_size=64),steps_per_epoch=len(grew_train_data) / 32,
         epochs=5)
