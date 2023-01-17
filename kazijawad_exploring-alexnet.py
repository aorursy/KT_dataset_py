from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras

# Load CIFAR dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Display data dimensions
print("Training Samples Shape:", x_train.shape)
print("Training Samples Count:", x_train.shape[0])
print("Test Samples Count:", x_test.shape[0])

# One-Hot Encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
l2_reg = 0

model = Sequential()

# 1st Convolution Layer
model.add(Conv2D(96, (11,11), input_shape=x_train.shape[1:], padding="same", kernel_regularizer=l2(l2_reg)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# 2nd Convolution Layer
model.add(Conv2D(256, (5,5), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# 3rd Convolution Layer
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# 4th Convolution Layer
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(1024, (3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))

# 5th Convolution Layer
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(1024, (3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# 1st Fully-Connected Layer
model.add(Flatten())
model.add(Dense(3072))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

# 2nd Fully-Connected Layer
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))

# 3rd Fully-Connected Layer
model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation("softmax"))

print(model.summary())
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adadelta(), metrics=["accuracy"])
# Training Parameters
batch_size = 32
epochs = 1

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
model.save("./cifar10_AlexNet.h5")

# Evaluate Performance
scores = model.evaluate(x_test, y_test, verbose=1)
print("Test Loss:", scores[0])
print("Test Accuracy:", scores[1])