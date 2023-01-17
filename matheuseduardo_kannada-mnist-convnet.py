import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from keras import Model, backend as K

from keras.layers import *

from keras.optimizers import SGD

from keras.callbacks import ModelCheckpoint, LambdaCallback

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical



from sklearn.model_selection import train_test_split
epochs = 30

batch_size = 128

dropout = .5

alpha = 1 # Variable used to decay dropout and data augmentation during training.



df_train = pd.read_csv('../input/Kannada-MNIST/train.csv')

df_test = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')



x_train, y_train = np.array(df_train.iloc[:, 1:]), np.array([i for i in df_train.iloc[:, 0].apply(to_categorical, args=(10,))])

# x_test, y_test = np.array(df_test.iloc[:, 1:]), np.array([i for i in df_test.iloc[:, 0].apply(to_categorical, args=(10,))])

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=.3, random_state=1234)



# Reshape images and map values between -1 and 1.

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))

x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))



del df_train, df_test
class VariableDropout(Layer):

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):

        self.rate = K.variable(min(1., max(0., rate)))

        self.noise_shape = noise_shape

        self.seed = seed

        super(VariableDropout, self).__init__(**kwargs)



    def call(self, x, training=None):

        def dropout():

            return K.dropout(x,

                             self.rate,

                             self._get_noise_shape(x),

                             seed=self.seed)

            

        return K.in_train_phase(dropout, x,

                                training=training)

    

    def compute_output_shape(self, input_shape):

        return input_shape

        

    def set_rate(self, rate):

        K.set_value(self.rate, min(1, max(0., rate)))

        

    def _get_noise_shape(self, inputs):

        if self.noise_shape is None:

            return self.noise_shape



        symbolic_shape = K.shape(inputs)

        noise_shape = [symbolic_shape[axis] if shape is None else shape

                       for axis, shape in enumerate(self.noise_shape)]

        

        return tuple(noise_shape)
def model():

    x = Input((28, 28, 1))

    

    y = Conv2D(32, 3, activation='relu')(x)

    y = BatchNormalization()(y)

    y = Conv2D(32, 4, strides=2, activation='relu')(y)

    y = BatchNormalization()(y)

    

    y = Conv2D(64, 3, activation='relu')(y)

    y = BatchNormalization()(y)

    y = Conv2D(64, 4, strides=2, activation='relu')(y)    

    y = BatchNormalization()(y)

    

    y = Conv2D(128, 3, activation='relu')(y)

    y = BatchNormalization()(y)

    y = Conv2D(128, 2, activation='relu')(y)

    y = BatchNormalization()(y)

    

    y = Flatten()(y)

    y = VariableDropout(dropout, seed=1234, name='dropout')(y)

    y = Dense(256, activation='relu')(y)

    y = Dense(10, activation='softmax')(y)

    

    return Model(x, y)



model = model()

model.compile(SGD(momentum=.9, nesterov=True), 'categorical_crossentropy', metrics=['acc'])

model.summary()
train_idg = ImageDataGenerator(rotation_range=.25*alpha,

                               width_shift_range=.2*alpha,

                               height_shift_range=.2*alpha,

                               zoom_range=.2*alpha,

                               preprocessing_function=lambda x: x/127.5-1)

train_idg = train_idg.flow(x_train,

                           y_train,

                           batch_size,

                           shuffle=False,

                           seed=1234)

    

test_idg = ImageDataGenerator(preprocessing_function=lambda x: x/127.5-1)

test_idg = test_idg.flow(x_test,

                         y_test,

                         batch_size=batch_size,

                         shuffle=False)
def adjustAlpha(epoch, logs):

    global model, alpha, dropout

    alpha *= .9

    model.get_layer('dropout').set_rate(dropout*alpha)

    

adjust_alpha = LambdaCallback(on_epoch_end=adjustAlpha)

model_checkpoint = ModelCheckpoint('Weights.h5', 'val_acc', save_best_only=True, save_weights_only=True, mode='max')
history = model.fit_generator(train_idg,

                              len(train_idg),

                              epochs,

                              callbacks=[adjust_alpha, model_checkpoint],

                              validation_data=test_idg,

                              validation_steps=len(test_idg),

                              shuffle=False).history
print('Best accuracy: %.2f' % (max(history['val_acc'])*100), '%', sep='')



plt.figure(figsize=(8, 6))

plt.plot(history['loss'], 'b-', label='loss')

plt.plot(history['val_loss'], 'r-', label='val_loss')

plt.title('Losses')

plt.legend()

plt.show()



plt.figure(figsize=(8, 6))

plt.plot(history['acc'], 'b-', label='acc')

plt.plot(history['val_acc'], 'r-', label='val_acc')

plt.title('Accuracies')

plt.legend()

plt.show()
model.load_weights('Weights.h5')

sample_submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

df_test = pd.read_csv('../input/Kannada-MNIST/test.csv')

df_test = np.array(df_test.iloc[:, 1:]).reshape((df_test.shape[0], 28, 28, 1))/127.5-1
for i in range(df_test.shape[0]):

    sample_submission['label'][i] = np.argmax(model.predict(df_test[i:i+1]))

sample_submission.to_csv('submission.csv', index=False)