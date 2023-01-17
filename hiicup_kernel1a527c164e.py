from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers import Dense
# Initialising the CNN

model = Sequential()



# Step 1 - Convolution

model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(rate = 0.3))



# Adding a second convolutional layer

model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(rate = 0.3))



# Step 3 - Flattening

model.add(Flatten())



# Step 4 - Full connection

model.add(Dense(units = 128, activation = 'relu'))

model.add(Dense(units = 1, activation = 'sigmoid'))



# Compiling the CNN

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



model.summary()
#Data Augmentation to prevent Overfitting

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory(directory='../input/Convolutional_Neural_Networks/dataset/training_set',

                                                 target_size = (64, 64),

                                                 batch_size = 32,

                                                 class_mode = 'binary')



test_set = test_datagen.flow_from_directory(directory='../input/Convolutional_Neural_Networks/dataset/test_set',

                                            target_size = (64, 64),

                                            batch_size = 32,

                                            class_mode = 'binary')
model.fit_generator(training_set,

                         steps_per_epoch = 1000,

                         epochs = 20,

                         validation_data = test_set,

                         validation_steps = 1000)
model.fit_generator(training_set,

                         steps_per_epoch = 2000,

                         epochs = 5,

                         validation_data = test_set,

                         validation_steps = 1000)