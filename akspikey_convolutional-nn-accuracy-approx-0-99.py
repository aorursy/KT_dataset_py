from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt

import numpy as np

from keras.preprocessing import image
classifier = Sequential()
classifier.add(Convolution2D(32, (3, 3), input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32, (3, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))

classifier.add(Dense(1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory('../input/pothole-images/Archive/training_set',

                                                target_size=(64, 64),

                                                batch_size=32,

                                                class_mode='binary')



test_set = test_datagen.flow_from_directory('../input/pothole-images/Archive/test_set',

                                            target_size=(64, 64),

                                            batch_size=32,

                                            class_mode='binary')
classifier.fit_generator(training_set,

        steps_per_epoch=562,

        epochs=25,

        validation_data=test_set,

        validation_steps=101)
plt.plot(classifier.history.history['accuracy'])

plt.plot(classifier.history.history['val_accuracy'])

plt.title('Analysis of the model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['Accuracy', 'Value Accuracy'], loc='upper right')

plt.show()
def predictImg(imgpath):

    predict_image = image.load_img(imgpath, target_size = (64,64))

    predict_image = image.img_to_array(predict_image)

    predict_image = np.expand_dims(predict_image, axis=0)

    result = classifier.predict(predict_image)

    return result.max()
predictImg('../input/pothole-image-prediction/prediction_data/road1.jpeg')
predictImg('../input/pothole-image-prediction/prediction_data/road2.jpg')
predictImg('../input/pothole-image-prediction/prediction_data/road3.jpg')
predictImg('../input/pothole-image-prediction/prediction_data/road4.jpeg')
predictImg('../input/pothole-image-prediction/prediction_data/road5.jpg')