# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import keras

from keras.datasets import cifar10

from keras.utils import np_utils

from keras.models import Model

from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization

from keras.applications.vgg16 import VGG16

from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score

from keras.preprocessing.image import ImageDataGenerator
LABEL_NAMES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']



VALIDATION_SIZE = 10000
(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()
LABEL_NAMES[y_train_all[1][0]]
x_train_all = x_train_all /255.0

x_test = x_test /255.0 
x_val = x_train_all[:VALIDATION_SIZE]

y_val = y_train_all[:VALIDATION_SIZE]

x_val.shape
x_train = x_train_all[VALIDATION_SIZE:]

y_train = y_train_all[VALIDATION_SIZE:]

x_train.shape
# Convert to One Hot Encoding

y_train_ohe = np_utils.to_categorical(y_train, num_classes=10)

y_test_ohe = np_utils.to_categorical(y_test, num_classes=10)

y_val_ohe = np_utils.to_categorical(y_val, num_classes=10)



print(y_val_ohe)

y_val_ohe.shape


def create_cnn_model():

    image_input = Input(shape=(32, 32, 3))

    

    vgg_model  = VGG16(weights='imagenet',include_top=False, input_tensor=image_input)

    

    flatt = Flatten()(vgg_model.output)

    

    couche1 = Dense(128, activation='relu')(flatt) 

    couche1_normalization = BatchNormalization()(couche1)

    couche1_dropout = Dropout(0.2)(couche1_normalization)

    couche2 = Dense(64, activation='relu')(couche1_dropout)

    couche2_normalization = BatchNormalization()(couche2)

    output = Dense(10, activation='softmax', name='output')(couche2_normalization)     

    model = Model( image_input, output )

    return model



model = create_cnn_model()

model.summary()


model.compile(optimizer='adam', 

              loss='categorical_crossentropy',

                metrics=['accuracy'])
# Use Data Augmentation

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip= True)

es = EarlyStopping(patience=10, monitor='val_accuracy', mode='max')

mc = ModelCheckpoint('./weights.h5', monitor='val_accuracy', mode='max', save_best_only=True)



model.fit_generator(datagen.flow(x_train, y_train_ohe,batch_size = 32), steps_per_epoch = 1250, epochs=500, validation_data=[x_val, y_val_ohe], callbacks = [es,mc])

# Load The Best weights in the ModelCheckpoint

model.load_weights('./weights.h5')



# Predict The Test

preds = model.predict(x_val)

score_test = accuracy_score( y_val, np.argmax(preds, axis=1) )



print (' LE SCORE DE TEST : ', score_test)

print('')
# after fit we can evaluate our model

_, evaluate = model.evaluate(x_test, y_test_ohe, verbose=1)

print('>%.3f' % (evaluate * 100.0))