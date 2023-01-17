# Load packages

import matplotlib.pyplot as plt

import numpy as np



from sklearn.model_selection import train_test_split

from tensorflow.keras import datasets, optimizers

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, DepthwiseConv2D, Flatten, Input, Layer, MaxPool2D

from tensorflow.keras.utils import plot_model
# Dataset loading

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()



# Get shape 

n_train, h, w = X_train.shape

n_test, _, _ = X_test.shape



# Reshape data

X_train = X_train[:, :, :, np.newaxis]

X_test = X_test[:, :, :, np.newaxis]



# Convert data

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255



# Split data

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,

                                                  test_size=10000,

                                                  random_state=42)



INPUT_SHAPE = (h, w, 1)
print(f'Shape of the train set: {X_train.shape}.')

print(f'Shape of the validation set: {X_val.shape}.')

print(f'Shape of the test set: {X_test.shape}.')
# Plot a random image

plt.imshow(X_train[42].squeeze(-1))

plt.title(f'Label: {y_train[42]}')

plt.axis('off')

plt.show()
print(f'There are {len(np.unique(y_train))} unique labels.')
# Define LeNet model

lenet = Sequential(name='LeNet-5')



lenet.add(Conv2D(6, (5, 5), padding='same', activation='tanh', input_shape=INPUT_SHAPE, name='C1'))

lenet.add(MaxPool2D(pool_size=(2, 2), name='S2'))

lenet.add(Conv2D(16, (5, 5), activation='tanh', name='C3'))

lenet.add(MaxPool2D(pool_size=(2, 2), name='S4'))

lenet.add(Conv2D(120, (5, 5), activation='tanh', name='C5'))

lenet.add(Flatten())

lenet.add(Dense(84, activation='tanh', name='F6'))

lenet.add(Dense(10, activation='softmax'))
plot_model(lenet, show_layer_names=False, show_shapes=True)
# Compile model

N_EPOCHS = 5

BATCH_SIZE = 256



lenet.compile(optimizer=optimizers.SGD(lr=0.1),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
# Fit model

history = lenet.fit(X_train, y_train,

                    epochs=N_EPOCHS,

                    batch_size=BATCH_SIZE,

                    validation_data=(X_val,y_val))
# Evaluate model on the test set

results_lenet = lenet.evaluate(X_test, y_test)
print(f'On test set: loss = {results_lenet[0]}, accuracy = {results_lenet[1]}.')
# Define Inception layer

def Inception(tensor, n_filters):

    """Define an Inception layer

    

    :param tensor: Instanciation of the layer

    :param n_filters: 

    :return: 

    """

    # Define the 4 branches

    branch1x1 = Conv2D(n_filters, kernel_size=(1, 1), activation='relu', padding='same')(tensor)

    branch3x3 = Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')(tensor)

    branch5x5 = Conv2D(n_filters, kernel_size=(5, 5), activation='relu', padding='same')(tensor)

    branch_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(tensor)

    

    # Merge the branches using Concatenate layer

    output = Concatenate(axis=-1)([branch1x1, branch3x3, branch5x5, branch_pool])

    return output
# Define the model

input_tensor = Input(shape=INPUT_SHAPE)

x = Conv2D(16, kernel_size=(5, 5), padding='same')(input_tensor)

x = Inception(x, 32)

x = Flatten()(x)

output_tensor = Dense(10, activation='softmax')(x)

inception_model = Model(inputs=input_tensor, outputs=output_tensor)
plot_model(inception_model, show_layer_names=False, show_shapes=True)
# Compile model

N_EPOCHS = 5

BATCH_SIZE = 256



inception_model.compile(optimizer=optimizers.SGD(lr=0.1),

                        loss='sparse_categorical_crossentropy',

                        metrics=['accuracy'])
# Fit model

history = inception_model.fit(X_train, y_train,

                              epochs=N_EPOCHS,

                              batch_size=BATCH_SIZE,

                              validation_data=(X_val,y_val))
# Evaluate model on the test set

results_inception = inception_model.evaluate(X_test, y_test)
print(f'On test set: loss = {results_inception[0]}, accuracy = {results_inception[1]}.')
# Define Residual block

class ResidualBlock(Layer):

    def __init__(self, n_filters):

        super().__init__(name='ResidualBlock')

        

        self.conv1 = Conv2D(n_filters, kernel_size=(3, 3), activation='relu', padding='same')

        self.conv2 = Conv2D(n_filters, kernel_size=(3, 3), padding='same')

        self.add = Add()

        self.last_relu = Activation('relu')

        

    def call(self, inputs):

        x = self.conv1(inputs)

        x = self.conv2(x)

        

        y = self.add([x, inputs])

        return self.last_relu(y)

    

# Define ResNet model

class MiniResNet(Model):

    def __init__(self, n_filters):

        super().__init__()

        

        self.conv = Conv2D(n_filters, kernel_size=(5, 5), padding='same')

        self.block = ResidualBlock(n_filters)

        self.flatten = Flatten()

        self.classifier = Dense(10, activation='softmax')

        

    def call(self, inputs):

        x = self.conv(inputs)

        x = self.block(x)

        x = self.flatten(x)



        return self.classifier(x)
# Define model

resnet_model = MiniResNet(32)

resnet_model.build((None, *INPUT_SHAPE))
# The plot_model function is not available with Oriented-Object API

resnet_model.summary()
# Compile model

N_EPOCHS = 5

BATCH_SIZE = 256



resnet_model.compile(optimizer=optimizers.SGD(lr=0.1),

                     loss='sparse_categorical_crossentropy',

                     metrics=['accuracy'])
# Fit model

history = resnet_model.fit(X_train, y_train,

                           epochs=N_EPOCHS,

                           batch_size=BATCH_SIZE,

                           validation_data=(X_val,y_val))
# Evaluate model on the test set

results_resnet = resnet_model.evaluate(X_test, y_test)
print(f'On test set: loss = {results_resnet[0]}, accuracy = {results_resnet[1]}.')
# Define Convolution block with BatchNorm

class ConvBlock(Layer):

    def __init__(self, n_filters, kernel_size, padding):

        super().__init__(name='ConvBlock')

        

        self.conv = Conv2D(n_filters, kernel_size=kernel_size, padding=padding, use_bias=False)

        self.bn = BatchNormalization(axis=3)

        self.activation = Activation('relu')

        

    def call(self, inputs):

        return self.activation(self.bn(self.conv(inputs)))



# Redefine Residual block with BatchNorm

class ResidualNormBlock(Layer):

    def __init__(self, n_filters):

        super().__init__(name='ResidualNormBlock')



        self.conv1 = ConvBlock(n_filters, kernel_size=(3, 3), padding='same')

        self.conv2 = ConvBlock(n_filters, kernel_size=(3, 3), padding='same')

        self.add = Add()

        self.last_relu = Activation('relu')

        

    def call(self, inputs):

        x = self.conv1(inputs)

        x = self.conv2(x)

        

        y = self.add([x, inputs])

        return self.last_relu(y)
# Define the model

input_tensor = Input(shape=INPUT_SHAPE)

x = Conv2D(32, kernel_size=(5, 5), padding='same')(input_tensor)

x = ResidualNormBlock(32)(x)

x = Flatten()(x)

output_tensor = Dense(10, activation='softmax')(x)

resnet_norm_model = Model(inputs=input_tensor, outputs=output_tensor)
plot_model(resnet_norm_model, show_layer_names=False, show_shapes=True)
# Compile model

N_EPOCHS = 5

BATCH_SIZE = 256



resnet_norm_model.compile(optimizer=optimizers.SGD(lr=0.1),

                          loss='sparse_categorical_crossentropy',

                          metrics=['accuracy'])
# Fit model

history = resnet_norm_model.fit(X_train, y_train,

                                epochs=N_EPOCHS,

                                batch_size=BATCH_SIZE,

                                validation_data=(X_val,y_val))
# Evaluate model on the test set

results_resnet_norm = resnet_norm_model.evaluate(X_test, y_test)
print(f'On test set: loss = {results_resnet_norm[0]}, accuracy = {results_resnet_norm[1]}.')
# Define Convolution model

conv_model = Sequential(name='ConvModel')

conv_model.add(Conv2D(8, kernel_size=(3, 3), use_bias=False))
# Build model

conv_model.build((None, *INPUT_SHAPE))

conv_model.summary()
# Define Separable Convolution model

separable_model = Sequential(name='SeparableModel')

separable_model.add(DepthwiseConv2D(kernel_size=(3, 3), use_bias=False))

separable_model.add(Conv2D(8, kernel_size=(1, 1), use_bias=False))
# Build model

separable_model.build((None, *INPUT_SHAPE))

separable_model.summary()