# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory (/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train)

TRAIN_DIR = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train'

VAL_DIR = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val'

TEST_DIR = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test'
# Part 1 - Building the CNN



# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense



# Initialising the CNN

classifier = Sequential()



# Step 1 and 2 - Convolution and Pooling

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional layer

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))





classifier.summary()
# Since the classes are not balanced, we cannot use accuracy as a  metric to analyze the model performance

# Metrics F1, precision, and recall have been removed from Keras. So we will use a custom metric function:



from keras import backend as K



def F1(y_true, y_pred):

    

    def precision(y_true, y_pred):

        """ Batch-wise average precision calculation



        Calculated as tp / (tp + fp), i.e. how many selected items are relevant

        Added epsilon to avoid the Division by 0 exception

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    

    def recall(y_true, y_pred):

        """ Batch-wise average recall calculation



        Computes the Recall metric, or Sensitivity or True Positive Rate  

        Calculates as tp / (tp + fn) i.e. how many relevant items are selected



        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall

   

    

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*(precision*recall)/(precision+recall+K.epsilon())







# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [F1])



# Part 2 - Fitting the CNN to the images



from keras.preprocessing.image import ImageDataGenerator

batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory(TRAIN_DIR,

                                                 target_size = (64, 64),

                                                 batch_size = batch_size,

                                                 class_mode = 'binary')



valid_set = test_datagen.flow_from_directory(VAL_DIR,

                                            target_size = (64, 64),

                                            batch_size = batch_size,

                                            class_mode = 'binary')







#  The instance of ImageDataGenerator().flow_from_directory(...) has an attribute 'filenames' 

# which is a list of all the files in the order the generator yields them

n_training_files = len(training_set.filenames)

n_valid_files = len(valid_set.filenames)
# steps_per_epoch parameter: the number of batches of samples it will take to complete one full epoch

# should be equivalent to the total number of samples divided by the batch size.



classifier.fit_generator(training_set,

                         steps_per_epoch = n_training_files/batch_size,

                         epochs = 25,

                         validation_data = valid_set,

                         validation_steps = n_valid_files/batch_size)
test_set = test_datagen.flow_from_directory(TEST_DIR,

                                            target_size = (64, 64),

                                            batch_size = batch_size,

                                            class_mode = 'binary')



test_accuracy = classifier.evaluate_generator(test_set,steps=624)

print('The testing accuracy is :', test_accuracy[1]*100, '%')
