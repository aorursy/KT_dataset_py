import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, AveragePooling2D, Activation, ELU, BatchNormalization, Layer, Lambda, LeakyReLU

from keras.optimizers import Adadelta, Adam, RMSprop

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.datasets.mnist import load_data as load

from keras.models import load_model

from keras.regularizers import l2

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras import backend as K

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')
x = train.iloc[:, 1:].values.astype('float32') / 255

y = train.iloc[:, 0]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2 , random_state=1337)
x_train = x_train.reshape(-1, 28, 28,1)

x_validation = x_validation.reshape(-1, 28, 28,1)

y_train = to_categorical(y_train)

y_validation = to_categorical(y_validation)
def build_model():

    model = Sequential()



    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=(28,28,1)))

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(ELU(alpha=1.0))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.4))

    

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(ELU(alpha=1.0))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.4))

    

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))

    model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(ELU(alpha=1.0))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.4))



    model.add(Flatten())

    model.add(Dense(512))

    model.add(BatchNormalization())

    model.add(ELU(alpha=1.0))

    model.add(Dropout(0.25))

    model.add(Dense(10, activation='softmax'))

    

    return model
data_gen = ImageDataGenerator(

        featurewise_center=False,

        samplewise_center=False,

        featurewise_std_normalization=False, 

        samplewise_std_normalization=False,

        zca_whitening=False,  

        rotation_range=10,  

        zoom_range = 0.1,

        shear_range=0.3,

        width_shift_range=0.1,

        height_shift_range=0.1, 

        horizontal_flip=False, 

        vertical_flip=False)  



data_gen.fit(x_train)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



es = EarlyStopping(monitor='val_loss',

                   min_delta=1e-1, 

                   verbose=1, 

                   patience=100,

                   restore_best_weights=True)
optimizer = RMSprop(learning_rate=0.002,

    rho=0.9,

    epsilon=1e-07,)
model = build_model() 

model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
batch_size = 1024

epochs = 50
history = model.fit_generator(data_gen.flow(x_train, y_train, batch_size=batch_size), 

                              validation_data=data_gen.flow(x_validation, y_validation, batch_size=batch_size), 

                              steps_per_epoch=len(x_train)//batch_size, 

                              epochs=epochs, 

                              callbacks=[learning_rate_reduction, es])
raw_test = pd.read_csv('../input/Kannada-MNIST/test.csv')
sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

raw_test_id=raw_test.id

raw_test=raw_test.drop("id",axis="columns")

raw_test=raw_test / 255

test=raw_test.values.reshape(-1,28,28,1)



sub=model.predict(test)     

sub=np.argmax(sub,axis=1) 



sample_sub['label']=sub

sample_sub.to_csv('submission.csv',index=False)