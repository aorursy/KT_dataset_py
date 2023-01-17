import keras
from keras.applications import VGG19
from keras.layers import Dropout
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import regularizers
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Activation,Flatten,MaxPool2D,Conv2D,Dropout
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import itertools
%matplotlib inline
from keras.datasets import cifar10
(X_train,y_train),(X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                          horizontal_flip=True)


it_train = datagen.flow(X_train, y_train, batch_size=32)
model = VGG19(
    weights=None,
    classes=10,
    input_shape=[32,32,3]
)
model.summary()
#model is unchanged
x = model.layers[-4].output

x=Dense(512, activation='relu')(x) 
x=BatchNormalization()(x)
x=Dropout(0.2)(x)
x=Dense(256, activation='relu')(x)
x=BatchNormalization()(x)
x=Dense(256, activation='relu')(x)
x=Dropout(0.3)(x)
x=Dense(256, activation='relu')(x)
x=Dense(128, activation='relu')(x)
x=Dense(128, activation='relu')(x)
x=Dropout(0.5)(x)
prediction_layer = Dense(10, activation='sigmoid')(x) 
model = Model(inputs=model.input, outputs=prediction_layer)

model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)
import os

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'VGG19'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
from tensorflow.keras.callbacks import ModelCheckpoint

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

callbacks = [checkpoint]
# Train the model
h = model.fit(
    x=X_train,
    y=y_train,
    validation_split=0.1,
    batch_size=32,
    epochs=20,
    callbacks=callbacks
)
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
model.save('VGG19.h5')
model.save_weights('VGG19_w.hdf5')
import pickle

f=open('VGG19_h.pckl','wb')
pickle.dump(h.history,f)
f.close()
import matplotlib.pyplot as plt
epoch_nums = range(1, 21)
training_loss = h.history["loss"]
validation_loss = h.history["val_loss"]
plt.plot(epoch_nums , training_loss)
plt.plot(epoch_nums , validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training','validation'], loc='upper right')
plt.show()