import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
import keras
def show_head(dataframe):
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(dataframe[i], cmap=plt.cm.binary)
    plt.show()
train_pure = np.load('../input/train_images_pure.npy')
show_head(train_pure)
train_noisy = np.load('../input/train_images_noisy.npy')
show_head(train_noisy)
train_rotated = np.load('../input/train_images_rotated.npy')
show_head(train_rotated)
train_both = np.load('../input/train_images_both.npy')
show_head(train_both)
testX = np.load('../input/Test_images.npy')
show_head(testX)
from keras.utils import np_utils
trainY = pd.read_csv("../input/train_labels.csv")
print(trainY['label'].value_counts(), '\n\n')
trainY = np.array(trainY.drop(['Id'], axis=1))
trainY = np_utils.to_categorical(trainY)
print(trainY)
train_pure = train_pure / 255
train_noisy = train_noisy / 255
train_rotated = train_rotated / 255
train_both = train_both / 255
testX = testX / 255

print (train_pure[0])
train_pure = train_both.reshape(train_both.shape[0], 1, 28, 28).astype('float32')
train_noisy = train_noisy.reshape(train_both.shape[0], 1, 28, 28).astype('float32')
train_rotated = train_rotated.reshape(train_rotated.shape[0], 1, 28, 28).astype('float32')
train_both = train_both.reshape(train_both.shape[0], 1, 28, 28).astype('float32')
testX = testX.reshape(testX.shape[0], 1, 28, 28).astype('float32')
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=gpu_options))
sess.close()
from keras.callbacks import TensorBoard
tensorboard = TensorBoard()
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
def SequentialModel():
    sequential_model = Sequential()
    #Input Layer
    sequential_model.add(Flatten())
    #Hidden Layers
    sequential_model.add(Dense(284))
    sequential_model.add(LeakyReLU(alpha=0.2))

    sequential_model.add(Dense(128, activation='sigmoid'))

    sequential_model.add(Dense(128, activation='sigmoid'))

    sequential_model.add(Dropout(0.2)) # Para evitar overfitting

    #Output Layer
    sequential_model.add(Dense(10)) 
    sequential_model.add(Activation('softmax'))

    sequential_model.compile(optimizer='sgd',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
    return sequential_model

sequential_model = SequentialModel()
history = sequential_model.fit(train_pure, trainY, epochs=12, validation_split=0.1)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
sequential_model.summary()
# Base com ruidos
evalu = sequential_model.evaluate(train_noisy[:6000], trainY[:6000], batch_size=32, verbose=1)
print ("loss: ", evalu[0], "   accuracy: ", evalu[1])
# Base rotacionada
evalu = sequential_model.evaluate(train_rotated[:6000], trainY[:6000], batch_size=32, verbose=1)
print ("loss: ", evalu[0], "   accuracy: ", evalu[1])
# Base rotacionada com ruidos
evalu = sequential_model.evaluate(train_both[:6000], trainY[:6000], batch_size=32, verbose=1)
print ("loss: ", evalu[0], "   accuracy: ", evalu[1])
sequential_model = SequentialModel()
history = sequential_model.fit(train_noisy, trainY, epochs=12, validation_split=0.1)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
import cv2
import numpy as np
train_noisy_gauss = np.zeros((60000, 1, 28, 28))
train_both_gauss = np.zeros((60000, 1, 28, 28))
for i in range(len(train_noisy)):
    no = cv2.GaussianBlur(train_noisy[i], (3, 3), 0)
    train_noisy_gauss[i] = no
    
for i in range(len(train_both)):
    no = cv2.GaussianBlur(train_both[i], (3, 3), 0)
    train_both_gauss[i] = no

image = train_both_gauss[0].reshape(28,28)
plt.imshow(image, cmap=plt.cm.binary)
print(train_both_gauss)
sequential_model = SequentialModel()
history = sequential_model.fit(train_noisy_gauss, trainY, epochs=12, validation_split=0.1)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from keras.layers import Conv2D
def conv_model():
    model = Sequential()


    model.add(Conv2D(10, (3,3), input_shape = (1, 28, 28), data_format='channels_first'))
    model.add(Activation("sigmoid"))
    
    model.add(Conv2D(10, (3,3), data_format='channels_first'))
    model.add(Activation("sigmoid"))


    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(20))
    model.add(LeakyReLU(alpha=0.3))
    

    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", #categorical_crossentropy
                optimizer = "adam",
                metrics = ["accuracy"])
    return model
conv_model().summary()
conv = conv_model()
history = conv.fit(train_rotated,
         trainY,
         validation_split=0.1,
         batch_size=32,
         epochs=3)


print(history.history.keys())


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
conv = conv_model()
history = conv.fit(train_both,
         trainY,
         validation_split=0.1,
         batch_size=32,
         epochs=3)


print(history.history.keys())


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
conv = conv_model()
history = conv.fit(train_both_gauss,
         trainY,
         validation_split=0.1,
         epochs=3)

print(history.history.keys())


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from keras.layers import MaxPooling2D
def dcnn(seed):
    conv = Sequential()


    conv.add(Conv2D(10, (3,3), input_shape = (1, 28, 28), data_format='channels_first'))
    conv.add(Activation("relu"))
    conv.add(MaxPooling2D(2,2))
    
    conv.add(Conv2D(20, (3,3), data_format='channels_first'))
    conv.add(Activation("relu"))
    conv.add(MaxPooling2D(2,2))


    conv.add(Flatten())
    conv.add(Dropout(0.2))
    conv.add(Dense(36))
    conv.add(LeakyReLU(alpha=0.3))
    

    conv.add(Dense(10))
    conv.add(Activation("softmax"))

    conv.compile(loss="categorical_crossentropy", #categorical_crossentropy
                optimizer = "adam",
                metrics = ["accuracy"])
    return conv
dcnn().summary()
dcnn = dcnn(7)
history = dcnn.fit(train_rotated,
         trainY,
         validation_split=0.1,
         batch_size=32,
         epochs=3)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
dcnn = dcnn()
history = dcnn.fit(train_both,
         trainY,
         validation_split=0.1,
         batch_size=32,
         epochs=3)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
pure_eval = dcnn.evaluate(train_pure[:12000], trainY[:12000], batch_size=32, verbose=1)
noisy_eval = dcnn.evaluate(train_noisy[:12000], trainY[:12000], batch_size=32, verbose=1)
noisyg_eval = dcnn.evaluate(train_noisy_gauss[:12000], trainY[:12000], batch_size=32, verbose=1)

print("Pure Training Dataset:")
print ("loss: ", pure_eval[0], "   accuracy: ", pure_eval[1])
print("Noisy Training Dataset:")
print ("loss: ", noisyg_eval[0], "   accuracy: ", noisy_eval[1])
print("Noisy Training Dataset with Gaussian Blur:")
print ("loss: ", noisyg_eval[0], "   accuracy: ", noisy_eval[1])
