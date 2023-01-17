from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
# Initializing CNN

classifier = Sequential()



# Adding Feature Detection

classifier.add(Conv2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))



# Adding Max Pooling

classifier.add(MaxPooling2D(pool_size = (2,2)))



# Flattening

classifier.add(Flatten())



# Adding connections

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))



# Compiling

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images



from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/train',

                                                 target_size = (32, 32),

                                                 batch_size = 20,

                                                 class_mode = 'binary')



test_set = test_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/test',

                                            target_size = (32, 32),

                                            batch_size = 20,

                                            class_mode = 'binary')



classifier.fit_generator(training_set,

                         samples_per_epoch = 5219,

                         nb_epoch = 5,

                         validation_data = test_set,

                         nb_val_samples = 625)
import numpy as np

from keras.preprocessing import image

test_image = image.load_img('/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg', target_size = (32,32))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)

print(training_set.class_indices)

print(result)
# importing Image class from PIL package 

from PIL import Image



# creating a object  

Image.open("/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg")  
