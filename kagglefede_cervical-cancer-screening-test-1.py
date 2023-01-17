

from subprocess import check_output

print(check_output(["ls", "../input/test"]).decode("utf8"))



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K



# dimensions of our images.

img_width, img_height = 150, 150

train_data_dir = '../input/train'

validation_data_dir = '../input/test'

nb_train_samples = 2000

nb_validation_samples = 800

epochs = 50

batch_size = 16



if K.image_dim_ordering() == 'channels_first':

    input_shape2 = (3, img_width, img_height)

else:

    input_shape2 = (img_width, img_height, 3)

    

model = Sequential()

model.add(Conv2D(32, 3, 3,  input_shape=input_shape2))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, 3, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, 3, 3))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))



model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation('softmax'))



model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1./255)



# this is a generator that will read pictures found in

# subfolers of 'data/train', and indefinitely generate

# batches of augmented image data

train_generator = train_datagen.flow_from_directory(

        train_data_dir,  # this is the target directory

        target_size=(150, 150),  # all images will be resized to 150x150

        batch_size=batch_size,

        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels



# this is a similar generator, for validation data

validation_generator = test_datagen.flow_from_directory(

        validation_data_dir,

        target_size=(150, 150),

        batch_size=batch_size,

        class_mode='binary')



model.fit_generator(

        train_generator,

        nb_epoch = 20,

        samples_per_epoch = 1480//20,

        nb_val_samples = 1480,

        validation_data=validation_generator)

model.save_weights('first_try.h5')  # always save your weights after training or during training


