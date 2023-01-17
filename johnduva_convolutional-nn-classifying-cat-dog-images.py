from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator
classifier = Sequential()
classifier.add(Convolution2D(32, (3, 3), 

                      input_shape=(64,64,3),

                      activation='relu'))

#help(Convolution2D)
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(activation = 'relu', 

                    units = 128,))
classifier.add(Dense(activation = 'sigmoid',

                    units = 1))
classifier.compile(optimizer = 'adam',

                   loss = 'binary_crossentropy',

                   metrics = ['accuracy'])
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory(

        '../input/training_set/training_set',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')



test_set = validation_generator = test_datagen.flow_from_directory(

        '../input/test_set/test_set',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')
classifier.fit_generator(

        training_set,

        steps_per_epoch=8000,

        epochs=5,

        validation_data=test_set,

        validation_steps=2000)