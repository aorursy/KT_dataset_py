import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline

import os

import gc
print(os.listdir("../input"))
data_fer = pd.read_csv('../input/fer2013/fer2013.csv')

data_fer.head()
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

idx_to_emotion_fer = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
X_fer_train, y_fer_train = np.rollaxis(data_fer[data_fer.Usage == "Training"][["pixels", "emotion"]].values, -1)

X_fer_train = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_train]).reshape((-1, 48, 48))

y_fer_train = y_fer_train.astype('int8')



X_fer_test_public, y_fer_test_public = np.rollaxis(data_fer[data_fer.Usage == "PublicTest"][["pixels", "emotion"]].values, -1)

X_fer_test_public = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_test_public]).reshape((-1, 48, 48))

y_fer_test_public = y_fer_test_public.astype('int8')



X_fer_test_private, y_fer_test_private = np.rollaxis(data_fer[data_fer.Usage == "PrivateTest"][["pixels", "emotion"]].values, -1)

X_fer_test_private = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_test_private]).reshape((-1, 48, 48))

y_fer_test_private = y_fer_test_private.astype('int8')
print(f"X_fer_train shape: {X_fer_train.shape}; y_fer_train shape: {y_fer_train.shape}")

print(f"X_fer_test_public shape: {X_fer_test_public.shape}; y_fer_test_public shape: {y_fer_test_public.shape}")

print(f"X_fer_test_private shape: {X_fer_test_private.shape}; y_fer_test_private shape: {y_fer_test_private.shape}")
plt.imshow(X_fer_train[10], interpolation='none', cmap='gray')

plt.title(idx_to_emotion_fer[y_fer_train[10]])

plt.show()

plt.imshow(X_fer_test_public[10], interpolation='none', cmap='gray')

plt.title(idx_to_emotion_fer[y_fer_test_public[10]])

plt.show()

plt.imshow(X_fer_test_private[10], interpolation='none', cmap='gray')

plt.title(idx_to_emotion_fer[y_fer_test_private[10]])

plt.show()
from keras.applications import VGG16

from keras.models import Model, Sequential

from keras.layers import Flatten, Dense, Input, Concatenate

from keras.utils import to_categorical
def one_hot(y):

    return to_categorical(y, 7)
conv_base = VGG16(weights=None, include_top=False, 

                    input_shape=(48, 48, 3))



img_input = Input(shape=(48,48,1))

img_conc = Concatenate()([img_input, img_input, img_input])   

conv_output = conv_base(img_conc)



conv_output_flattened = Flatten()(conv_output)

dense_out = Dense(128, activation='relu')(conv_output_flattened)

out = Dense(7, activation='softmax')(dense_out)



model = Model(inputs=img_input, outputs=out)



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy']) 





model.fit(

    X_fer_train.reshape((-1, 48, 48, 1)), 

    one_hot(y_fer_train), 

    batch_size=128, 

    epochs=15, 

    validation_data=(X_fer_test_public.reshape((-1, 48, 48, 1)), one_hot(y_fer_test_public)))
from keras.layers import Dense , Activation

from keras.layers import Dropout

from keras.layers import Flatten

from keras.constraints import maxnorm

from keras.optimizers import SGD , Adam

from keras.layers import Conv2D , BatchNormalization

from keras.layers import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K



def swish_activation(x):

    return (K.sigmoid(x) * x)
model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(48,48, 1)))

model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))

model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))

model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))

model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation=swish_activation))

model.add(Dropout(0.4))

model.add(Dense(7 , activation='sigmoid'))



model.compile(loss='categorical_crossentropy',

              optimizer='adam' ,

              metrics=['categorical_accuracy'])



print(model.summary())
model.fit(

    X_fer_train.reshape((-1, 48, 48, 1)), 

    one_hot(y_fer_train), 

    batch_size=128, 

    epochs=15, 

    validation_data=(X_fer_test_public.reshape((-1, 48, 48, 1)), one_hot(y_fer_test_public)))
!pip install git+https://github.com/rcmalli/keras-vggface.git

from keras_vggface.vggface import VGGFace

from keras_vggface import utils
VGGFace(include_top = False, input_shape = (48,48,3),pooling = 'avg').summary()
vggfeatures = VGGFace(include_top = False, input_shape = (48,48,3),pooling = 'avg')

for x in vggfeatures.layers[:]:

    x.trainable = False

base_model = vggfeatures



img_input = Input(shape=(48,48,1))

img_conc = Concatenate()([img_input, img_input, img_input])   

conv_output = base_model(img_conc)



dense_out = Dense(128, activation='relu')(conv_output)

out = Dense(7, activation='softmax')(dense_out)



model = Model(inputs=img_input, outputs=out)



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy'])



model.fit(

    X_fer_train.reshape((-1, 48, 48, 1)), 

    one_hot(y_fer_train), 

    batch_size=128, 

    epochs=15, 

    validation_data=(X_fer_test_public.reshape((-1, 48, 48, 1)), one_hot(y_fer_test_public)))
vggfeatures = VGGFace(include_top = False, input_shape = (48,48,3),pooling = 'avg')

for x in vggfeatures.layers[:-5]:

    x.trainable = False

base_model = vggfeatures



img_input = Input(shape=(48,48,1))

img_conc = Concatenate()([img_input, img_input, img_input])   

conv_output = base_model(img_conc)



dense_out = Dense(128, activation='relu')(conv_output)

out = Dense(7, activation='softmax')(dense_out)



model = Model(inputs=img_input, outputs=out)



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy'])



model.fit(

    X_fer_train.reshape((-1, 48, 48, 1)), 

    one_hot(y_fer_train), 

    batch_size=128, 

    epochs=15, 

    validation_data=(X_fer_test_public.reshape((-1, 48, 48, 1)), one_hot(y_fer_test_public)))
vggfeatures = VGGFace(include_top = False, input_shape = (48,48,3),pooling = 'avg')



base_model = vggfeatures



img_input = Input(shape=(48,48,1))

img_conc = Concatenate()([img_input, img_input, img_input])   

conv_output = base_model(img_conc)



dense_out = Dense(128, activation='relu')(conv_output)

out = Dense(7, activation='softmax')(dense_out)



model = Model(inputs=img_input, outputs=out)



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy'])



model.fit(

    X_fer_train.reshape((-1, 48, 48, 1)), 

    one_hot(y_fer_train), 

    batch_size=128, 

    epochs=15, 

    validation_data=(X_fer_test_public.reshape((-1, 48, 48, 1)), one_hot(y_fer_test_public)))
conv_base = VGG16(weights='imagenet', include_top=False, 

                    input_shape=(48, 48, 3))

conv_base.trainable = False



img_input = Input(shape=(48,48,1))

img_conc = Concatenate()([img_input, img_input, img_input])   

conv_output = conv_base(img_conc)



conv_output_flattened = Flatten()(conv_output)

dense_out = Dense(128, activation='relu')(conv_output_flattened)

out = Dense(7, activation='softmax')(dense_out)



model = Model(inputs=img_input, outputs=out)



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy']) 



model.fit(

    X_fer_train.reshape((-1, 48, 48, 1)), 

    one_hot(y_fer_train), 

    batch_size=128, 

    epochs=15, 

    validation_data=(X_fer_test_public.reshape((-1, 48, 48, 1)), one_hot(y_fer_test_public)))
conv_base = VGG16(weights='imagenet', include_top=False, 

                    input_shape=(48, 48, 3))





for x in conv_base.layers[:-4]:

    x.trainable = False



img_input = Input(shape=(48,48,1))

img_conc = Concatenate()([img_input, img_input, img_input])   

conv_output = conv_base(img_conc)



conv_output_flattened = Flatten()(conv_output)

dense_out = Dense(128, activation='relu')(conv_output_flattened)

out = Dense(7, activation='softmax')(dense_out)



model = Model(inputs=img_input, outputs=out)



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy']) 





model.fit(

    X_fer_train.reshape((-1, 48, 48, 1)), 

    one_hot(y_fer_train), 

    batch_size=128, 

    epochs=15, 

    validation_data=(X_fer_test_public.reshape((-1, 48, 48, 1)), one_hot(y_fer_test_public)))
conv_base = VGG16(weights='imagenet', include_top=False, 

                    input_shape=(48, 48, 3))



img_input = Input(shape=(48,48,1))

img_conc = Concatenate()([img_input, img_input, img_input])   

conv_output = conv_base(img_conc)



conv_output_flattened = Flatten()(conv_output)

dense_out = Dense(128, activation='relu')(conv_output_flattened)

out = Dense(7, activation='softmax')(dense_out)



model = Model(inputs=img_input, outputs=out)



model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy']) 





model.fit(

    X_fer_train.reshape((-1, 48, 48, 1)), 

    one_hot(y_fer_train), 

    batch_size=128, 

    epochs=15, 

    validation_data=(X_fer_test_public.reshape((-1, 48, 48, 1)), one_hot(y_fer_test_public)))