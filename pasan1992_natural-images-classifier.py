from keras.preprocessing import image

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator



imageSize = 64



datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.6,

        zoom_range=0.6,

        horizontal_flip=True,

        validation_split=0.2

        )



train_generator = datagen.flow_from_directory(

        '/kaggle/input/natural-images/data/natural_images/',

        target_size=(imageSize, imageSize),

        batch_size=64,

        class_mode='categorical',

        subset='training')



test_generator = datagen.flow_from_directory(

        '/kaggle/input/natural-images/data/natural_images/',

        target_size=(imageSize, imageSize),

        batch_size=64,

        class_mode='categorical',

        subset='validation')
batch_size = 64



model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', input_shape=(imageSize,imageSize,3), activation="relu"))

model.add(Conv2D(32, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))

model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(8, activation="softmax"))



LR_SCHEDULE = [(5, 0.001),(15, 0.0005),(20, 0.0001),(25, 0.00001)]



def lr_schedule(epoch, lr):

  if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:

    print(lr)

    return lr

  for i in range(len(LR_SCHEDULE)):

    if epoch == LR_SCHEDULE[i][0]:

      print(lr)

      return LR_SCHEDULE[i][1]

    

    print(lr)

  return lr



reduce_lr = LearningRateScheduler(lr_schedule)



model.compile(

    loss='categorical_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)



epochs = 50

steps_per_epoch = train_generator.n // batch_size

validation_steps = test_generator.n // batch_size



history = model.fit_generator(train_generator,

                              steps_per_epoch = steps_per_epoch,

                              epochs=epochs,

                              workers=4,

                              validation_data=test_generator,

                              validation_steps=validation_steps,

                              callbacks=[reduce_lr])