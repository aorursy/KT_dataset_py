#Importing required libraries.

from keras.models import Sequential

# To initialize neural network

from keras.layers import Convolution2D

# Images are two dimensional, concolution step

from keras.layers import MaxPooling2D

# Pooling step

from keras.layers import Flatten

# Convert pools feature map into this large feature vector

from keras.layers import Dense

#To add fully connected layers
#Initializing the CNN

#There is also a graph option but we'll use sequential ANN Model

classifier = Sequential()



#step 1 - Convolution

#creating the feature map by using feature detector from ınput image



classifier.add( Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))

#32 Feature maps&detetctors uses 3 by 3 matrices, we can put 128 in the powerful machines

#step -2 Pooling

classifier.add(MaxPooling2D(pool_size = (2,2)))



#second convolution and pooling steps.

classifier.add( Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))



classifier.add(MaxPooling2D(pool_size = (2,2)))
#step -3 Flattening

classifier.add(Flatten())
#step-4 Full connection step

classifier.add(Dense(output_dim = 256, activation = 'relu'))

classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#binary outcome
#compiling the cnn



classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
#Fitting to CNN to the images



from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)





test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory(

        '/kaggle/input/chestdataset/dataset2/two/train',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')



test_set = test_datagen.flow_from_directory(

        '/kaggle/input/chestdataset/dataset2/two/test',

        target_size=(64, 64),

        batch_size=32,

        class_mode='binary')

#We'll try to train with 100 Epochs

results=classifier.fit_generator(

        training_set,

        samples_per_epoch=130,

        nb_epoch=100,

        validation_data=test_set,

        nb_val_samples=18,verbose=1)
import matplotlib.pyplot as plt

def plot_acc_loss(results, epochs):

 acc = results.history['accuracy']

 loss = results.history['loss']

 val_acc = results.history['val_accuracy']

 val_loss = results.history['val_loss']

 plt.figure(figsize=(15, 5))

 plt.subplot(121)

 plt.plot(range(1,epochs), acc[1:], label='Train_acc')

 plt.plot(range(1,epochs), val_acc[1:], label='Test_acc')

 plt.title('Accuracy over ' + str(epochs) + ' Epochs', size=15)

 plt.legend()

 plt.grid(True)

 plt.subplot(122)

 plt.plot(range(1,epochs), loss[1:], label='Train_loss')

 plt.plot(range(1,epochs), val_loss[1:], label='Test_loss')

 plt.title('Loss over ' + str(epochs) +  ' Epochs', size=15)

 plt.legend()

 plt.grid(True)

 plt.show()

 

plot_acc_loss(results, 100)
# Part 3 - Making new predictions

# Testing

import numpy as np

from keras.preprocessing import image



#First learn the classification indices.

print(training_set.class_indices)

%pylab inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



img=mpimg.imread('/kaggle/input/chestdataset/dataset2/two/single_prediction/covid.jpeg')

imgplot = plt.imshow(img)



test_image = image.load_img('/kaggle/input/chestdataset/dataset2/two/single_prediction/covid.jpeg', target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)



if result[0][0] == 1:

    prediction = 'normal'

else:

    prediction = 'covid'

    

#print("AI's prediction is: "+ prediction)



plt=plt.title('Prediction is  '+ prediction )

#There we will test this following image, COVID-19 positive
%pylab inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



test_image = image.load_img('/kaggle/input/chestdataset/dataset2/two/single_prediction/covid2.jpeg', target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)



img=mpimg.imread('/kaggle/input/chestdataset/dataset2/two/single_prediction/covid2.jpeg')

imgplot = plt.imshow(img)



if result[0][0] == 1:

    prediction = 'normal'

else:

    prediction = 'covid'

    

#print("AI's prediction is: "+ prediction)

plt=plt.title('Prediction is  '+ prediction )



#There we will test this following image, COVID-19 positive
%pylab inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



test_image = image.load_img('/kaggle/input/chestdataset/dataset2/two/single_prediction/normal.jpeg', target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = classifier.predict(test_image)



img=mpimg.imread('/kaggle/input/chestdataset/dataset2/two/single_prediction/normal.jpeg')

imgplot = plt.imshow(img)



if result[0][0] == 1:

    prediction = 'normal'

else:

    prediction = 'covid'

    

#print("AI's prediction is: "+ prediction)



plt=plt.title('Prediction is  '+ prediction )