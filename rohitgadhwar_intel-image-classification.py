# Importing the Keras libraries and packages

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2

from sklearn.utils import shuffle

from random import randint

from IPython.display import SVG

import matplotlib.gridspec as gridspec

import tensorflow.keras.utils as Utils

from keras.utils.vis_utils import model_to_dot

import numpy as np

import os

import matplotlib.pyplot as plot

from sklearn.metrics import confusion_matrix as CM
def get_images(directory):

    Images = []

    Labels = []  # 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street

    label = 0

    

    for labels in os.listdir(directory): #Main Directory where each class label is present as folder name.

        if labels == 'glacier': #Folder contain Glacier Images get the '2' class label.

            label = 2

        elif labels == 'sea':

            label = 4

        elif labels == 'buildings':

            label = 0

        elif labels == 'forest':

            label = 1

        elif labels == 'street':

            label = 5

        elif labels == 'mountain':

            label = 3

        

        for image_file in os.listdir(directory+labels): #Extracting the file name of the image from Class Label folder

            image = cv2.imread(directory+labels+r'/'+image_file) #Reading the image (OpenCV)

            image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)

            Images.append(image)

            Labels.append(label)

    

    return shuffle(Images,Labels,random_state=817328462) #Shuffle the dataset you just prepare
def get_classlabel(class_code):

    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}

    

    return labels[class_code]
Images, Labels = get_images('../input/intel-image-classification/seg_train/seg_train/') #Extract the training images from the folders.



Images = np.array(Images) #converting the list of images to numpy array.

Labels = np.array(Labels)

print("Shape of Images:",Images.shape)

print("Shape of Labels:",Labels.shape)
f,ax = plot.subplots(5,5) 

f.subplots_adjust(0,0,3,3)

for i in range(0,5,1):

    for j in range(0,5,1):

        rnd_number = randint(0,len(Images))

        ax[i,j].imshow(Images[rnd_number])

        ax[i,j].set_title(get_classlabel(Labels[rnd_number]))

        ax[i,j].axis('off')

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('../input/intel-image-classification/seg_train/seg_train',

                                                 target_size = (150, 150),

                                                 batch_size = 32,

                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('../input/intel-image-classification/seg_test/seg_test', 

                                            target_size = (150, 150),

                                            batch_size = 32,

                                            class_mode = 'categorical')
# Initialising the CNN

classifier = Sequential()



# Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))



# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional layer

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(128, activation = 'relu'))

classifier.add(Dense(6, activation = 'softmax'))



# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.summary()
trained = classifier.fit(training_set,

                         steps_per_epoch = 14034//32,

                         epochs = 15,

                         validation_data = test_set,

                         validation_steps = 3000//32)
plot.plot(trained.history['accuracy'])

plot.plot(trained.history['val_accuracy'])

plot.title('Model accuracy')

plot.ylabel('Accuracy')

plot.xlabel('Epoch')

plot.legend(['Train', 'Test'], loc='upper left')

plot.show()
plot.plot(trained.history['loss'])

plot.plot(trained.history['val_loss'])

plot.title('Model loss')

plot.ylabel('Loss')

plot.xlabel('Epoch')

plot.legend(['Train', 'Test'], loc='upper left')

plot.show()

pred_images,no_labels = get_images('../input/intel-image-classification/seg_pred/')

pred_images = np.array(pred_images)

pred_images.shape
fig = plot.figure(figsize=(30, 30))

outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)



for i in range(25):

    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    rnd_number = randint(0,len(pred_images))

    pred_image = np.array([pred_images[rnd_number]])

    pred_class = get_classlabel(classifier.predict_classes(pred_image)[0])

    pred_prob = classifier.predict(pred_image).reshape(6)

    for j in range(2):

        if (j%2) == 0:

            ax = plot.Subplot(fig, inner[j])

            ax.imshow(pred_image[0])

            ax.set_title(pred_class)

            ax.set_xticks([])

            ax.set_yticks([])

            fig.add_subplot(ax)

        else:

            ax = plot.Subplot(fig, inner[j])

            ax.bar([0,1,2,3,4,5],pred_prob)

            fig.add_subplot(ax)





fig.show()