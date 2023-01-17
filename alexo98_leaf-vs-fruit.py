# Aroung 4,280,514 total parameters





# Using a pretrained network (MobileNet - it is lightweight and it is good enough)

# Using a pretrained network has on average 8% more accuracy than using a 



# Removed every class that doesn't have a disease associated with it

# 100 pixels images

# Image segmentation with a lot of zooming, rotating and shifting



# In this version I decided to remove the validation after the training , and instead take 20% of the whole

# dataset and validate it at each step (This does not affect our training, as the model does not learn from the

# validation, it only applies a .predict() function)

# The only downside is that we can't have a conffusion matrix



# Sigmoid as an activation layer

from keras.applications.mobilenet import MobileNet, preprocess_input



COLOR = 'RGB'

IMG_SIZE = 128

BATCHES = 32

EPOCHS = 30

VERSION = 16





if COLOR == 'GRAYSCALE':

    CHANNELS = 1

else:

    CHANNELS = 3

import os

import cv2

import numpy as np

import sys # So I can remove the printing limit for the arrays

import gc

import pandas as pd

from blist import blist # A list library that is more efficient in terms of memory than a list and is faster than an np.array

import matplotlib.pyplot as plt 

import keras

from tqdm  import tqdm, tqdm_notebook # A library which prints a progress bar for loops



from keras import regularizers

from keras.models import Model

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation,AveragePooling2D, GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD,Adam,RMSprop

from keras.models import Sequential, load_model

from keras.callbacks import History

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical





from collections import Counter 



from sklearn.metrics import confusion_matrix

from sklearn.utils import shuffle 



from sklearn.model_selection import train_test_split





# PATH = '../input/leaf-v-fruit/type classification/Type classification/'

# PATH = '../input/leaf-v-other/leaf/'

# PATH = '../input/flower-type/flower type recognition/Flower Type Recognition/Train'

# PATH = '../input/ftype3/flower type recognition 2/Flower Type Recognition/Train'

PATH = '../input/individual-disease/disease class/'

# PATH_TEST = '../input/ftype3/flower type recognition 2/Flower Type Recognition/Test'

NUM_CLASSES=len(os.listdir(PATH))



CATEGORIES = []



model_name = 'plants_v'+str(VERSION)+'.model'



print(CHANNELS)

# Functions for printing the loss and accuracy for each epoch



def print_data_validation(history):

    print()

    print("Validation")

    print()

    for i in range(len(history.history['val_loss'])):

        val_loss = float("{0:.4f}".format(history.history['val_loss'][i]))

        val_acc = float("{0:.4f}".format(history.history['val_acc'][i]))

        print("Epoch "+str(i+1)+": Loss:",val_loss," Accuracy: ",val_acc)

def print_data_training(history):

    print()

    print("Training")

    print()

    for i in range(len(history.history['loss'])):

        loss = float("{0:.4f}".format(history.history['loss'][i]))

        acc = float("{0:.4f}".format(history.history['acc'][i]))

        print("Epoch "+str(i+1)+": Loss:",loss," Accuracy: ",acc)
def create_model(_path,_epochs,_batch):

    



    

    train_datagen = ImageDataGenerator(rescale=1./255,

                                       validation_split = 0.1,                                       

                                       horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(_path,

                                                        target_size=(IMG_SIZE, IMG_SIZE),

                                                        batch_size=_batch,

                                                        color_mode= COLOR.lower(),

                                                        class_mode='categorical',

                                                        subset='training')

    

    test_generator = train_datagen.flow_from_directory(_path,

                                                      target_size=(IMG_SIZE,IMG_SIZE),

                                                      batch_size=_batch,

                                                      color_mode= COLOR.lower(),

                                                      class_mode='categorical',

                                                      subset='validation')

    

    np.save('categories'+str(VERSION)+'.npy', train_generator.class_indices)

    

    # Loading the pretrained network (MobileNet)

    # It is the most lightweigth pretrained network available for free online



    model = Sequential()  

    

    model.add(Conv2D(64, (4,4), padding='same', input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)))

    model.add(Activation('relu'))

    model.add(Conv2D(64, (4,4), padding='same'))

    model.add(Activation('relu'))

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.3))



    model.add(Conv2D(128, (3,3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.3))



    model.add(Conv2D(192, (3,3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.4))



    model.add(Conv2D(256, (2,2)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D((2,2)))

    model.add(Dropout(0.5))



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    # Output Layer

    model.add(Dense(NUM_CLASSES))

    model.add(Activation('softmax'))

#     model.summary()



    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    history = model.fit_generator(train_generator,

                                  steps_per_epoch = train_generator.n // _batch,

                                  epochs = _epochs,

                                  validation_data=test_generator,

                                  validation_steps=test_generator.n // _batch,

                                  verbose=1)







    

    model.save(model_name)

    

    # Plotting the model history (Accuracy and Loss over epochs)

    

    plt.plot(history.history['loss'])

    plt.plot(history.history['acc'])

    plt.title('Model history')

    plt.ylabel('Loss / Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Loss', 'Accuracy'], loc='upper left')

    plt.savefig('accuracy_plot.png')

    

    return model,history
model,history = create_model(PATH,EPOCHS,BATCHES)


print_data_validation(history)

print_data_training(history)
# def load_testing_data(path,color,_shuffle):

#     x = blist([])

#     y = blist([])

#     main_path=os.listdir(path)

#     main_path.sort()

#     CATEGORIES = init_categories()

#     for category in tqdm_notebook(main_path):  

#         k=0

#         new_path = os.path.join(path,category)

#         cat = CATEGORIES.index(category)

        

#         images_path = os.listdir(new_path)

#         images_path.sort()

        

#         for img in images_path:

#             try:                                

#                 if color == 'RGB':

#                     image = cv2.imread(os.path.join(new_path,img))

#                     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#                 elif color == 'GRAYSCALE':

#                     image = cv2.imread(os.path.join(new_path,img),cv2.IMREAD_GRAYSCALE)

#                 else:

#                     image = cv2.imread(os.path.join(new_path,img))

#                     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#                 img = None

#                 del img

                

#                 image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))             

#                 if path == PATH_TEST:

#                     image = image/255.0

#                 x.append(image)

#                 y.append(cat)

            

#             except Exception as e:

#                 print("Error: ",e)

      

#     if _shuffle:

#         x,y = shuffle(x,y)

#     return x,y



# def find_2nd_max(array,x):

#     max = x

#     snd_max = 0

#     index = 0

#     for i in range(len(array)):

#         if array[i]>snd_max and array[i]<max:

#             snd_max = array[i]

#             index = i

#     return index



# def print_confusion_matrix(y_test,y_pred,categ,acc,file):

#     conf = confusion_matrix(y_test,y_pred)

#     for i in range(len(conf)):

#         total = np.sum(conf[i])

        

#         max = conf[i][np.argmax(conf[i])]

#         snd_index = find_2nd_max(conf[i],max)

#         snd_max = conf[i][snd_index]

        

#         max_proc = float("{0:.3f}".format((max/total)*100))

#         snd_max_proc = float("{0:.3f}".format((snd_max/total)*100))

#         s = ""

#         s+="{0} ({1}): {2} ({3}%) ~~ {4}: {5} ({6}%)\n".\

#               format(categ[i],total,max,max_proc,categ[snd_index],snd_max,snd_max_proc)

#         file.write(s)

#         print(s)



        

# def predict():

#     res = model.predict(x_test)

#     CATEGORIES=init_categories()

#     no_correct = 0

#     no_wrong = 0

#     y_pred = []

#     for i in range(len(res)):

#         y_pred.append(np.argmax(res[i]))

#         if y_test[i] == np.argmax(res[i]):    

#             no_correct+=1

#         else:

#             no_wrong+=1

#     acc = float("{0:.3f}".format((no_correct/(no_correct+no_wrong))*100))

#     print("Correct: {} from {} ({}%)".format(no_correct,no_correct+no_wrong,acc))



#     file = open((str(VERSION)+'accuracy.txt'),'w')

#     print_confusion_matrix(y_test, y_pred,CATEGORIES,acc,file)

#     file.close()

# def init_categories():

#     path = os.listdir(PATH_TEST)

#     path.sort()

#     temp_cat = []

#     for x in path:

#             temp_cat.append(x)

    

#     temp_cat.sort()

    

#     return temp_cat
# CATEGORIES=[]

# x_test,y_test  = load_testing_data(PATH_TEST,COLOR,True)

# x_test = np.asarray(x_test)

# x_test = np.reshape(x_test,(-1,IMG_SIZE,IMG_SIZE,CHANNELS))

# predict()