# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
# Our data is located in three folders:

# train= contains the training data/images for teaching our model.



# val=    contains images which we will use to validate our model. The purpose of this data set is to prevent our model from Overfitting.

#         Overfitting is when your model gets a little too comfortable with the training data and can't handle data it hasn't see....too well.



# test = this contains the data that we use to test the model once it has learned the relationships between the images and their label (Pneumonia/Not-Pneumonia)
# Initializing the CNN



classifier = Sequential()



## Step 1 - Convolution

classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))



## Step 1 - Pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))



## Adding a second Convolution layer

classifier.add(Conv2D(32,(3,3),activation='relu'))



classifier.add(MaxPooling2D(pool_size=(2,2)))



# Flattening

classifier.add(Flatten())



# Full connection

#classifier.add(Dense(units=512,activation='relu'))

#classifier.add(Dense(units=256,activation='relu'))

classifier.add(Dense(units=128,activation='relu'))



## Adding a loss layer

classifier.add(Dense(units=1,activation='sigmoid'))

#binary output ..Pneumonia or Normal



# Compiling the CNN

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])





# Now, we are going to fit the model to our training dataset and we will keep out testing dataset separate 



# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator



train_model=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)



test_model=ImageDataGenerator(rescale=1./255)





train_set = train_model.flow_from_directory('../input/chest_xray/chest_xray/train',target_size=(64,64), batch_size=32, class_mode='binary')



validation_generator = test_model.flow_from_directory('../input/chest_xray/chest_xray/val', target_size=(64, 64), batch_size=32,

                                                        class_mode='binary')



test_set = test_model.flow_from_directory('../input/chest_xray/chest_xray/test',target_size=(64,64), batch_size=32, class_mode='binary')





classifier.summary()





classifier.fit_generator(train_set, steps_per_epoch=5216/32, epochs=10, validation_data = validation_generator, validation_steps=624/32)   



test_accu = classifier.evaluate_generator(test_set,steps=624)



print('The testing accuracy is :', test_accu[1]*100, '%')
import numpy as np  # for linear algebra

import matplotlib.pyplot as plt # for plotting graphs



#Accuracy

plt.plot(classifier.history.history['acc'])

plt.plot(classifier.history.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training_set', 'Validation_set'], loc='upper left')

plt.show()



# Loss 

plt.plot(classifier.history.history['val_loss'])

plt.plot(classifier.history.history['loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Test set'], loc='upper left')

plt.show()
#  Results



# After training, a separate test set was used to evaluate the performance of the CNN classifier. 

# The test set is a little unbalanced as 63% of the test chest x ray images are pneumonia.

# I also use this number as baseline accuracy. 

# The CNN classifier on Test_set achieved an accuracy of 89%, 

# which is substantially better than the baseline accuracy.



print('The testing accuracy is :', test_accu[1]*100, '%')



from keras.preprocessing import image

test_image = image.load_img('../input/chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg', target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)

print(result)

train_set.class_indices

print(train_set.class_indices)

if result[0][0] == 0:

    prediction = 'Normal'

    print(" The test image is")

    print(prediction)

else:

    prediction = 'Pneumonia'

    print(" The test image is")

    print(prediction)
