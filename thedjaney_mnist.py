from keras.layers import Flatten,Dense, Conv2D, Reshape, LeakyReLU, Dropout,MaxPooling2D
from keras.models import Model,Sequential
from keras.utils import to_categorical
# just get versions
from keras import __version__ as keras_version
from tensorflow import __version__ as tf_version
import numpy as np
import gzip

print('Keras: ',keras_version)
print('Tensorflow: ',tf_version)
# functions
import matplotlib.pyplot as plt  
import keras
def plot_history(history):
	plt.figure(1)  

	# summarize history for accuracy  

	plt.subplot(211)
	plt.plot(history.history['acc'])
	#plt.plot(history.history['val_acc'])
	plt.ylabel('accuracy')

	# summarize history for loss  

	plt.subplot(212)
	plt.plot(history.history['loss'])
	#plt.plot(history.history['val_loss'])  
	plt.ylabel('loss')  
	plt.xlabel('batch')  
	plt.show() 

class BatchLossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'acc': []}
        
    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('acc'))
        
def extract_images(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, 28, 28, 1)
    return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

x_train = extract_images('../input/train-images-idx3-ubyte.gz',60000)
y_train = extract_labels('../input/train-labels-idx1-ubyte.gz', 60000)
x_test = extract_images('../input/t10k-images-idx3-ubyte.gz',10000)
y_test = extract_labels('../input/t10k-labels-idx1-ubyte.gz', 10000)
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape labels to be categorical
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
model = Sequential()
model.add(Flatten(input_shape=(28,28,1)))
model.add(Dense(64, activation='relu',input_shape=(28,28)))
model.add(Dense(10, activation='softmax'))
model.compile('rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
## Summary
model.summary()
history = BatchLossHistory()
model.fit(x_train,y_train, validation_data=(x_test,y_test), callbacks=[history], batch_size=1000)
# chart
plot_history(history)
model = Sequential()
model.add(Conv2D(32, 3, padding='valid', activation=LeakyReLU(), input_shape=(28,28,1)))
model.add(Conv2D(64, 3, padding='valid', activation=LeakyReLU()))
model.add(MaxPooling2D(pool_size=2, strides=1, padding='valid'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(125, activation=LeakyReLU()))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile('rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
## Summary
model.summary()
history = BatchLossHistory()
model.fit(x_train,y_train, validation_data=(x_test,y_test), callbacks=[history], batch_size=1000, epochs=15)
# chart
plot_history(history)