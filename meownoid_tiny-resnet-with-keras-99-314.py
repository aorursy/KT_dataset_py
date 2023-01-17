import numpy as np                  # for working with tensors outside the network

import pandas as pd                 # for data reading and writing

import matplotlib.pyplot as plt     # for data inspection
from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout

from keras.layers.merge import add

from keras.activations import relu, softmax

from keras.models import Model

from keras import regularizers
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
def block(n_output, upscale=False):

    # n_output: number of feature maps in the block

    # upscale: should we use the 1x1 conv2d mapping for shortcut or not

    

    # keras functional api: return the function of type

    # Tensor -> Tensor

    def f(x):

        

        # H_l(x):

        # first pre-activation

        h = BatchNormalization()(x)

        h = Activation(relu)(h)

        # first convolution

        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)

        

        # second pre-activation

        h = BatchNormalization()(x)

        h = Activation(relu)(h)

        # second convolution

        h = Conv2D(kernel_size=3, filters=n_output, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(h)

        

        # f(x):

        if upscale:

            # 1x1 conv2d

            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)

        else:

            # identity

            f = x

        

        # F_l(x) = f(x) + H_l(x):

        return add([f, h])

    

    return f
# input tensor is the 28x28 grayscale image

input_tensor = Input((28, 28, 1))



# first conv2d with post-activation to transform the input data to some reasonable form

x = Conv2D(kernel_size=3, filters=16, strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(input_tensor)

x = BatchNormalization()(x)

x = Activation(relu)(x)



# F_1

x = block(16)(x)

# F_2

x = block(16)(x)



# F_3

# H_3 is the function from the tensor of size 28x28x16 to the the tensor of size 28x28x32

# and we can't add together tensors of inconsistent sizes, so we use upscale=True

# x = block(32, upscale=True)(x)       # !!! <------- Uncomment for local evaluation

# F_4

# x = block(32)(x)                     # !!! <------- Uncomment for local evaluation

# F_5

# x = block(32)(x)                     # !!! <------- Uncomment for local evaluation



# F_6

# x = block(48, upscale=True)(x)       # !!! <------- Uncomment for local evaluation

# F_7

# x = block(48)(x)                     # !!! <------- Uncomment for local evaluation



# last activation of the entire network's output

x = BatchNormalization()(x)

x = Activation(relu)(x)



# average pooling across the channels

# 28x28x48 -> 1x48

x = GlobalAveragePooling2D()(x)



# dropout for more robust learning

x = Dropout(0.2)(x)



# last softmax layer

x = Dense(units=10, kernel_regularizer=regularizers.l2(0.01))(x)

x = Activation(softmax)(x)



model = Model(inputs=input_tensor, outputs=x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
df_train = pd.read_csv('../input/train.csv')



y_train_ = df_train.ix[:, 0].values.astype(np.int).reshape(-1, 1)

x_train = df_train.ix[:, 1:].values.astype(np.float32).reshape((-1, 28, 28, 1))
df_test = pd.read_csv('../input/test.csv')



x_test = df_test.values.astype(np.float32).reshape((-1, 28, 28, 1))
y_train = OneHotEncoder(sparse=False).fit_transform(y_train_)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train_)
m = x_train.mean(axis=0)



x_train -= m

x_val -= m

x_test -= m
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
mc = ModelCheckpoint('weights.best.keras', monitor='val_acc', save_best_only=True)
def sigmoidal_decay(e, start=0, end=100, lr_start=1e-3, lr_end=1e-5):

    if e < start:

        return lr_start

    

    if e > end:

        return lr_end

    

    middle = (start + end) / 2

    s = lambda x: 1 / (1 + np.exp(-x))

    

    return s(13 * (-e + middle) / np.abs(end - start)) * np.abs(lr_start - lr_end) + lr_end
xs = np.linspace(0, 100)

ys = np.vectorize(sigmoidal_decay)(xs)

plt.plot(xs, ys)

plt.show()
EPOCHS = 3                        # !!! <------- Chnage to 30-100 for local evaluation
lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=EPOCHS))
hist = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val), batch_size=512, callbacks=[lr, mc])
loss = hist.history['loss']

val_loss = hist.history['val_loss']

epochs = np.arange(1, EPOCHS + 1)



plt.plot(epochs, loss)

plt.plot(epochs, val_loss)

plt.show()
acc = hist.history['acc']

val_acc = hist.history['val_acc']

epochs = np.arange(1, EPOCHS + 1)



plt.plot(epochs, acc)

plt.plot(epochs, val_acc)

plt.show()
model.load_weights('weights.best.keras')
p_test = model.predict(x_test, batch_size=512)

p_test = np.argmax(p_test, axis=1)
pd.DataFrame({'ImageId': 1 + np.arange(p_test.shape[0]), 'Label': p_test}).to_csv('output.csv', index=False)
model.load_weights('weights.best.keras')
p_test = model.predict(x_test, batch_size=512)

p_test = np.argmax(p_test, axis=1)
pd.DataFrame({'ImageId': 1 + np.arange(p_test.shape[0]), 'Label': p_test}).to_csv('output.csv', index=False)