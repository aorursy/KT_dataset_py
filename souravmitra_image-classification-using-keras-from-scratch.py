import numpy as np

import pandas as pd

from keras.models import Sequential      # Using the Sequential Model

from keras.layers import Convolution2D   # for the Convolution layer

from keras.layers import Activation

from keras.layers import MaxPooling2D    # for the Pooling layer

from keras.layers import Dense           # for the Fully Connected layer

from keras.layers import Flatten

from keras.layers import Dropout

from keras.preprocessing import image

import os
os.listdir('../input/cat-and-dog')
dog_train_path = []

cat_train_path = []



dog_test_path = []

cat_test_path = []



dog_train_folder_path = '../input/cat-and-dog/training_set/training_set/dogs'

cat_train_folder_path = '../input/cat-and-dog/training_set/training_set/cats'



dog_test_folder_path = '../input/cat-and-dog/test_set/test_set/dogs'

cat_test_folder_path = '../input/cat-and-dog/test_set/test_set/cats'



# importing the paths of dogs training set



for path in os.listdir(dog_train_folder_path) :

    if '.jpg' in path :

        dog_train_path.append(os.path.join(dog_train_folder_path, path))



# importing the paths of cats training set



for path in os.listdir(cat_train_folder_path) :

    if '.jpg' in path :

        cat_train_path.append(os.path.join(cat_train_folder_path, path))



# importing the paths of dogs testing set



for path in os.listdir(dog_test_folder_path) :

    if '.jpg' in path :

        dog_test_path.append(os.path.join(dog_test_folder_path, path))



# importing the paths of cats testing set 



for path in os.listdir(cat_test_folder_path) :

    if '.jpg' in path :

        cat_test_path.append(os.path.join(cat_test_folder_path, path))
print( len(dog_train_path), len(dog_test_path), len(cat_train_path), len(cat_test_path) )
training_data = np.zeros( (8000, 32, 32, 3), dtype = 'float32' )

training_label = []



testing_data = np.zeros( (2000, 32, 32, 3), dtype = 'float32' )

testing_label = []



# creating the training dataset



for i in range(8000) :

    if i < 4000 :

        img = image.load_img(dog_train_path[i], target_size = (32, 32))

        training_data[i] = image.img_to_array(img)

        training_label.append('dogs')

        

    else :

        img = image.load_img(cat_train_path[i - 4000], target_size = (32, 32))

        training_data[i] = image.img_to_array(img)

        training_label.append('cats')

        



# creating the testing data sets



for i in range(2000) :

    if i < 1000 :

        img = image.load_img(dog_test_path[i], target_size = (32, 32))

        testing_data[i] = image.img_to_array(img)

        testing_label.append('dogs')

        

    else :

        img = image.load_img(cat_test_path[i - 1000], target_size = (32, 32))

        testing_data[i] = image.img_to_array(img)

        testing_label.append('cats')

    

    
print( len(training_data), len(training_label), len(testing_data), len(testing_label) )
from sklearn.preprocessing import LabelEncoder



labelencoder = LabelEncoder()



training_labels = labelencoder.fit_transform(training_label)

testing_labels = labelencoder.fit_transform(testing_label)
model = Sequential()



model.add(Convolution2D( 32, 3, 3, input_shape = (32, 32, 3) ))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Convolution2D( 32, 3, 3, input_shape = (32, 32, 3) ))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Convolution2D( 32, 3, 3, input_shape = (32, 32, 3) ))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Dropout(0.5))



model.add(Flatten())



model.add(Dense(128))

model.add(Activation('relu'))



model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])



model.summary()
from keras.preprocessing.image import ImageDataGenerator



imagedatagenerator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)



training_data_final = imagedatagenerator.fit(training_data)

testing_data_final = imagedatagenerator.fit(testing_data)
model.fit_generator(imagedatagenerator.flow(training_data, training_labels), steps_per_epoch = 8000,

                    epochs = 4, verbose = 4, shuffle = True, )
training_labels
predictions = model.predict_classes(testing_data)
predictions
from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(testing_labels, predictions))

confusionmatrix = print(confusion_matrix(testing_labels, predictions))
# put the path of the image to classify 



image_path = '../input/cat-and-dog/training_set/training_set/dogs/dog.215.jpg'
# function to get the image 



def getImage(path) :

    img = image.load_img(path)

    return img
def getImageClass(path) :

    image_temp = image.load_img(path, target_size = (32, 32))

    image1 = image.img_to_array(image_temp)

    image1 = np.expand_dims(image1, axis = 0)

    

    result = model.predict_classes(image1)

    print(result)

    

    if result == 1 :

        

        print(' This is an image of a Dog ')

        return getImage(path)

        

    else :

        getImage(path)

        print(' This is an image of a Cat')

        return getImage(path)
getImageClass(image_path)