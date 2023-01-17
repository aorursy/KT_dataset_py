# I just need this for displaying an image, nothing else
import matplotlib.pyplot as plt


# Working with files? import os module
import os

# need it for formatting img array to numpy array
import numpy as np

# for data augmentation, thank u Exun for informing me about this technique
from keras.preprocessing.image import ImageDataGenerator

# In predicting images after the model has trained, i need load img to convert it into the basic format it was in ImageDataGenerator
from keras.preprocessing.image import load_img

# converts image into array of each item containing the amount of pixels from the width of the image in which an item represents rgb colors
from keras.preprocessing.image import img_to_array


""" ------------------------------------------------------------- ImageDataGenerator() for augmentation---------------------------------------"""

"""-------------------------------------------------------------------       generator     --------------------------------------------"""
# creating a ImageDataGenerator class for creating same images, but with some minor edits to generate more data
train_datagen = ImageDataGenerator(rescale=1./255,
    # cuts a bit
    shear_range=0.2,
    
    # zooms on the picture a bit
    zoom_range=0.2,
    
    # flipping the image on horizontal axis is ok since it wont change the whole concept
    horizontal_flip=True,
                                  
    validation_split=0.2
    
    ) 


""" -------------------------------------------------------Generate Image for train folder images --------------------------------"""
train_generator = train_datagen.flow_from_directory(
    
    # path for training data
    '../input/intel-image-classification/seg_train/seg_train',
    
    # input layer is the shape to take in data of only (100,100), therefore, we set the target size
    target_size=(150,150),
    
    # I heard batch size of 128 is good for any categorical AI
    batch_size=128,
    
    # 2 classes - binary, 6 classes - categorical
    class_mode='categorical',
    
    # data split
    subset="training"
    )


"""----------------------------------------------------------------- Data Split - validation data generator -------------------------"""
# validation generator for good accuracy (again thanks exun for showing me data split)
validation_generator = train_datagen.flow_from_directory(
    
    #path for testing data
    '../input/intel-image-classification/seg_train/seg_train',
    
    # input layer is the shape to take in data of only (150,150), therefore, we set the target size
    target_size=(150,150),
    
    # I heard batch size of 128 is good for any categorical AI
    batch_size=128,
    
    # 2 classes - binary, 6 classes - categorical
    class_mode='categorical',
    
    # data split
    subset="validation"
    )

""" ------------------------------------------------------- Tools/Modules for Model network ( for accuracy) ------------------------------------------"""
# tools for making layers of the neural net 
from keras.layers import Dense

# class for the neural net
from keras.models import Sequential

# input layer
from keras.layers import Conv2D

# Featuring Technique - It selects the brighter parts of the image to as if to highlight important notes 
from keras.layers import MaxPool2D

# Helps in reshaping the the tensors to have shape equal to number of elements in tensor
from keras.layers import Flatten

# Dropout technique, disabling some neurons randomly for AI to get better.
from keras.layers import Dropout

# Overfitting technique for Batches
from keras.layers.normalization import BatchNormalization

# THE MODEL ( neural net )


""" -----------------------------------------------------------  Model architecture ------------------------------------------------"""
model = Sequential()

""" ----------------------------------------------------------- Process input data --------------------------------------------------"""


# Normalize the Batches properly
model.add(BatchNormalization())



# input layer
model.add(Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))

""" -----------------------------------------------------The Image conversion net (with neurons) -------------------------------------"""

# First layer
model.add(Conv2D(180,kernel_size=(3,3),activation='relu'))

# Focus on brighter parts of the image ( or keep features of the image)
model.add(MaxPool2D(5,5))

# Nets again
model.add(Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(150,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(120,kernel_size=(3,3),activation='relu'))

# Just nets for minimizing neurons gradually
model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))

# Focus on brighter parts of the image ( or keep features of the image) 
model.add(MaxPool2D(5,5))

model.add(Conv2D(50,kernel_size=(3,3),activation='relu'))

""" -----------------------------------------------------             Predicting nets                     -------------------------------------"""

# Flatten values for Dense nets
model.add(Flatten())

# Just nets
model.add(Dense(180,activation='relu'))

model.add(Dropout(0.6))

model.add(Dense(150,activation='relu'))
model.add(Dense(120,activation='relu'))

# Dropout for overfitting
model.add(Dropout(0.5))

model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))

# Final output layer ( 6 nodes for 6 classes)
model.add(Dense(6,activation='softmax'))

"""--------------------------------------------------------------  compile and train      -----------------------------------------------"""
# compile - optimizer - ADAM, loss - categorical_crossentrophy
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])


# TRAIN (also take history )
history = model.fit_generator(

# our ImageDataGenerator.flow_from_directory() train data is inserted
train_generator,

# shuffle images
shuffle = True,
# validation data generator inserted
validation_data = validation_generator, 

# epochs (kinda went overkill with it)
epochs = 1000
)
# created this function to properly load every image to img array of proper image dimensions and format of test
# with answers for to check accuracy
def load_images_from_folder(folder):
    images = []
    answers = []
    for filename1 in os.listdir(folder):
        for filename in os.listdir(os.path.join(folder,filename1)):
            img = load_img(os.path.join(folder,filename1,filename), target_size=(150, 150
                                                                                ))
            
            img = img_to_array(img)
            
            if img is not None:
                images.append(img)
                answers.append(filename1)
    return [images,answers]
# loading images from folder to predict
test,answers = load_images_from_folder('../input/intel-image-classification/seg_test/seg_test')
for i in range(len(answers)):
    if answers[i] == 'buildings':
        answers[i] = np.array([1,0,0,0,0,0])
    if answers[i] == 'forest':
        answers[i] = np.array([0,1,0,0,0,0])
    if answers[i] == 'glacier':
        answers[i] = np.array([0,0,1,0,0,0])
    if answers[i] == 'mountain':
        answers[i] = np.array([0,0,0,1,0,0])
    if answers[i] == 'sea':
        answers[i] = np.array([0,0,0,0,1,0])
    if answers[i] == 'street':
        answers[i] = np.array([0,0,0,0,0,1])
# predict these as classes
predictions = model.evaluate(np.array(test),np.array(answers))