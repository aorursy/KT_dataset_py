# import libraries

# we are not importing libraries used for csv files, as keras deals with all of these



from keras.models import Sequential

from keras.layers import Convolution2D #images are two dimensional. Videos are three dimenstional with time.

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense



#initialize the classifier CNN

classifier = Sequential() #Please note that there is another way to build a mode: Functional API.



#applying convolution operation --> build the convolutional layer

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#32, 3, 3 --> 32 filters with 3 x 3 for each filter. 

#start with 32 filters, and then create more layers with 64, 128, 256, etc

#expected format of the images.

# 256, 256, 3 --> 3 color channels (RGB), 256 x 256 pixels. But when using CPU, 3, 64, 64 --> due to computational limitation
#Max Pooling --> create a pooling layer

classifier.add(MaxPooling2D(pool_size = (2,2)))

# 2 x 2 size --> commonly used to keep much information.



#Flattening --> creating a long vector.

classifier.add(Flatten()) #no parameters needed.



#classic ANN - full connection

classifier.add(Dense(output_dim = 128, activation = 'relu'))

#common practice: number of hidden nodes between the number of input nodes and output nodes, and choose powers of 2

classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
#Data augmentation

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255, 

                                   shear_range = 0.2, 

                                   zoom_range = 0.2, 

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('../input/training_set', 

                                                    target_size = (64, 64), 

                                                    batch_size = 32,

                                                   class_mode = 'binary')

test_set = test_datagen.flow_from_directory('../input/test_set',

                                                target_size = (64, 64),

                                                 batch_size = 32, 

                                                 class_mode = 'binary')



classifier.fit_generator(training_set, 

                         samples_per_epoch = 8005, 

                        nb_epoch = 2, 

                        validation_data = test_set, 

                        nb_val_samples = 2025)