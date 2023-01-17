from IPython.display import display, Image
from matplotlib import animation
from IPython.display import HTML
Image(url='https://cdn-images-1.medium.com/max/600/1*GdxHFaUDbvTXJreKg3S8SQ.gif')
# Import the required libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Initializing the CNN
Classifier=Sequential()
# Adding layers into CNN architecture
# Convolution Step
Classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
# Pooling Step
Classifier.add(MaxPooling2D(pool_size=(2,2)))
#Flattening Step
Classifier.add(Flatten())
# Fully Connected Layer
Classifier.add(Dense(128,activation='relu'))
Classifier.add(Dense(1,activation='sigmoid'))
# Compiling the model
Classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Image Preprocessing Step
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/kaggle/input/dogs-cats-images/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('/kaggle/input/dogs-cats-images/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
Classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 2000)
