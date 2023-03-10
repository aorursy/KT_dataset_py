import jovian

#Extract data that is fetched in data.zip, Uncomment the cell below
#!unzip data.zip
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
# dimensions of our images.

img_width, img_height = 200, 200
train_data_dir = 'data/train'

validation_data_dir = 'data/validation'

nb_train_samples = 1914

nb_validation_samples = 182

epochs = 50

batch_size = 16
hyperparams = {

    'img_width' : 100,

    'img_height' : 100,

    'nb_train_samples': 1914,

    'nb_validation_samples': 182,

    'epochs': 50,

    'batch_size' : 16

}



jovian.log_hyperparams(hyperparams)
if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)
if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)



model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('sigmoid'))                #Added

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

# augmentation 

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



# augmentation 

# only rescaling

test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')
checkpoint = ModelCheckpoint("first1_cp.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=5)
history = model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=nb_validation_samples // batch_size)
model.save('firstimplementation.h5') 
jovian.log_metrics({

    'loss': 0.1651,

    'acc': 0.9288,

    'val_loss': 0.1705,

    'val_acc': 0.9318

})
jovian.log("Final Model")
import matplotlib.pyplot as plt

%matplotlib inline

# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
jovian.commit(artifacts=['firstimplementation.h5', 'data.zip'])