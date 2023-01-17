import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
train_file = "../input/train.csv"

test_file = "../input/test.csv"

output_file = "submission.csv"
raw_data = np.loadtxt(train_file, skiprows=1, dtype="int", delimiter=",")



x_train, x_val, y_train, y_val = train_test_split(

    raw_data[:,1:], raw_data[:,0], test_size=0.1)
fig, ax = plt.subplots(2, 1, figsize=(12,6))



ax[0].plot(x_train[0])

ax[0].set_title("784x1 Data")

ax[1].imshow(x_train[0].reshape(28,28), cmap='gray')

ax[1].set_title("28x28 Data")
x_train = x_train.reshape(-1, 28, 28, 1)

x_val = x_val.reshape(-1, 28, 28, 1)



x_train = x_train.astype("float32")/255.

x_val = x_val.astype("float32")/255.
y_train = to_categorical(y_train)

y_val = to_categorical(y_val)



print(y_train[0])
model = Sequential()



# Convolution Layers

model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))

model.add(BatchNormalization())



# MaxPooling

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



# Convolution

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))

model.add(BatchNormalization())



# MaxPooling

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



# Further Methods

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25)) # regularization method

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
datagen = ImageDataGenerator(

    zoom_range=0.1,

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1)
model.compile(loss='categorical_crossentropy', optimizer= Adam(lr=1e-4), metrics=['accuracy'])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),

                           steps_per_epoch=500,

                           epochs=25,

                           verbose=2,

                          validation_data=(x_val[:400,:], y_val[:400,:]),

                          callbacks=[annealer])
final_loss, final_accuracy = model.evaluate(x_val, y_val, verbose=0)

print("Final Loss: {0:.4f}, Final Accuracy: {1:.4f}".format(final_loss, final_accuracy))
plt.plot(hist.history['loss'], color='b', label='Loss')

plt.plot(hist.history['val_loss'], color='r', label = 'Valuation Loss')

plt.legend()

plt.show()



plt.plot(hist.history['acc'], color='b', label = 'Accuracy')

plt.plot(hist.history['val_acc'], color='r', label = 'Valuation Accuracy')

plt.legend()

plt.show()
y_hat = model.predict(x_val)

y_pred = np.argmax(y_hat, axis=1)

y_true = np.argmax(y_val, axis=1)



cm = confusion_matrix(y_true, y_pred)

print(cm)
testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')

x_test = testset.astype("float32")

x_test = testset.reshape(-1, 28, 28, 1)/255.
y_hat = model.predict(x_test, batch_size=64)
y_pred = np.argmax(y_hat, axis=1)
with open(output_file, 'w') as f:

    f.write('Image_ID,Label\n')

    for i in range(len(y_pred)):

        f.write("".join([str(i+1), ',', str(y_pred[i]),'\n']))