import numpy
import gzip
# Params for MNIST
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
# Extract the images
def extract_data(filename, num_images):
    """""
    Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data
def extract_labels(filename, num_images):
    #Extract the labels into a vector of int64 label IDs.
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        num_labels_data = len(labels)
        one_hot_encoding = numpy.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[numpy.arange(num_labels_data),labels] = 1
        one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding
train_data = extract_data('../input/train-images-idx3-ubyte.gz', 40000)
train_labels = extract_labels('../input/train-labels-idx1-ubyte.gz', 40000)
x_test = extract_data('../input/t10k-images-idx3-ubyte.gz', 1000)
y_test = extract_labels('../input/t10k-labels-idx1-ubyte.gz', 1000)
print(train_data.shape)
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
import tensorflow as tf
x_train,  x_val, y_train, y_val  = train_test_split(train_data, train_labels, test_size = 0.2)
n_train = len(y_train)
n_val = len(y_val)
#set hyperparamters
input_dim = 28
batch_size = 50

#IP => [CONV => RELU => BN => POOL]*2 =>[FC => RELU => BN => DO]*2 =>OP
model = models.Sequential()
model.add(layers.Conv2D(16, (3,3), input_shape = (input_dim, input_dim,1)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size = (2, 2)))

model.add(layers.Conv2D(32, (3,3)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size = (2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(20))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(10))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(10))
model.add(layers.Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = 'Adam',
              metrics = ['accuracy'])
model.summary()

"""
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
"""
history = model.fit(
    x_train, y_train, 
    steps_per_epoch = n_train // batch_size,
    epochs = 50,
    validation_data = (x_val, y_val),
    validation_steps= n_val // batch_size
)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')
y_pred = model.predict(x_test,verbose=1)
correct_prediction = np.equal(np.argmax(y_pred, 1), np.argmax(y_test, 1))
accuracy = correct_prediction.sum()/y_test.shape[0]
print("Accuracy on test set :{:.2%}".format(accuracy))