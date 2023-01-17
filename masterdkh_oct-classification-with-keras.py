from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.core import Dense, Flatten, Dropout

from keras import optimizers

import numpy as np

from keras.preprocessing import image

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (4, 4)))

classifier.add(Conv2D(512, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 4, activation = 'sigmoid'))

classifier.summary()
# Compiling the CNN



classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

  rescale = 1./255

  )

training_set = train_datagen.flow_from_directory('../input/oct2017/OCT2017 /train',

  target_size = (150, 150),

  batch_size = 40,

  classes=["DME","CNV","NORMAL","DRUSEN"])



validation_set = train_datagen.flow_from_directory('../input/oct2017/OCT2017 /val',

  target_size = (150, 150),

  batch_size = 40,

  classes=["DME","CNV","NORMAL","DRUSEN"])



test_set = train_datagen.flow_from_directory('../input/oct2017/OCT2017 /test',

  target_size = (150, 150),

  batch_size = 40,

  classes=["DME","CNV","NORMAL","DRUSEN"])
classifier.fit_generator(training_set,

  steps_per_epoch = 83484/40,

  epochs = 30)
classifier.evaluate_generator(test_set, steps = 10)
classifier.evaluate_generator(validation_set, steps = 10)
classifier.save('trained.h5')