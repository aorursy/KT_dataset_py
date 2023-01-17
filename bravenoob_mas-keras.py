# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load the extension and start TensorBoard

%load_ext tensorboard.notebook

%tensorboard --logdir logs
import time 



#let's also import the abstract base class for our callback

from tensorflow.keras.callbacks import Callback



#defining the callback

class TimerCallback(Callback):

    

    def __init__(self, maxExecutionTime):

        

# Arguments:

#     maxExecutionTime (number): Time in minutes. The model will keep training 

#                                until shortly before this limit

#                                (If you need safety, provide a time with a certain tolerance)

        

        self.maxExecutionTime = maxExecutionTime * 60

    

    

    #Keras will call this when training begins

    def on_train_begin(self, logs):

        self.startTime = time.time()

        self.longestTime = 0            #time taken by the longest epoch or batch

        self.lastTime = self.startTime  #time when the last trained epoch or batch was finished



    #this is our custom handler that will be used in place of the keras methods:

        #`on_batch_end(batch,logs)` or `on_epoch_end(epoch,logs)`

    def on_epoch_end(self, epoch, logs):

        

        currentTime      = time.time()                           

        self.elapsedTime = currentTime - self.startTime    #total time taken until now

        thisTime         = currentTime - self.lastTime     #time taken for the current epoch

                                                               #or batch to finish

        

        self.lastTime = currentTime

        

        #verifications will be made based on the longest epoch or batch

        if thisTime > self.longestTime:

            self.longestTime = thisTime

        

        

        #if the (assumed) time taken by the next epoch or batch is greater than the

            #remaining time, stop training

        remainingTime = self.maxExecutionTime - self.elapsedTime

        if remainingTime < self.longestTime:

            

            self.model.stop_training = True  #this tells Keras to not continue training

            print("\n\nTimerCallback: Finishing model training before it takes too much time. (Elapsed time: " + str(self.elapsedTime/60.) + " minutes )\n\n")
train_df = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv")
import tensorflow as tf

print(tf.__version__)
from tensorflow.keras.applications import NASNetLarge

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

from tensorflow.keras.optimizers import Adam, Nadam, Adamax



num_classes = 5

model = NASNetLarge(weights='imagenet', include_top=False, pooling='avg')



my_new_model = Sequential()

my_new_model.add(model)

my_new_model.add(Dense(512, activation="relu"))

my_new_model.add(Dropout(rate=0.25))

my_new_model.add(Dense(256, activation="relu"))

my_new_model.add(Dropout(rate=0.25))                 

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# optimizer

# descent optimizer (adam lr defaut = 0.001)

my_new_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])



print(my_new_model.summary())
layers = [(layer, layer.name, layer.trainable) for layer in my_new_model.layers]

pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])   
from tensorflow.keras.applications.nasnet import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.callbacks import Callback



image_size = 331

batch_size = 64

nb_epochs = 50



# steps_per_epoch: number of yields (batches) before a epoch is over

# ceil(num_samples / batch_size)

# epochs: Number of epochs to train the model. An epoch is an iteration over the entire data provided

# class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). 

#  This can be useful to tell the model to "pay more attention" to samples from an under-represented class.



shift = 0.2



#brightness_range=[0.8,1.2],

#zoom_range=[0.8,1.2],

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, zoom_range=[0.9,1.1],width_shift_range=0.2, 

                                       height_shift_range=0.2, horizontal_flip=True)



#liefert die trainingsdaten als iterator

# image aug funktioniert so, dass für den aktuellen batch die bilder geändert werden. Nicht dass es mehr Bilder gibt.

# Somit wird für jede Epoche mit anderen Bildenr trainiert.

train_generator = train_datagen.flow_from_directory(

    '../input/petfinder-images/petfinder_images/images/train',

    target_size=(image_size, image_size),

    batch_size=batch_size,

    class_mode='categorical')



# keine image augmenation für validierung

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



#liefert die Validationdaten als iterator

validation_generator = test_datagen.flow_from_directory(

    '../input/petfinder-images/petfinder_images/images/validation',

    target_size=(image_size, image_size),

    batch_size=batch_size,

    class_mode='categorical')



STEP_SIZE_TRAIN=math.ceil(train_generator.n//train_generator.batch_size)

STEP_SIZE_VALID=math.ceil(validation_generator.n//validation_generator.batch_size)



print(STEP_SIZE_TRAIN)

print(STEP_SIZE_VALID)





# Configure the TensorBoard callback and fit the model

tensorboard_callback = TensorBoard("logs")



# Early stopping against overfit

earlystopping_callback = EarlyStopping(patience=batch_size/10, monitor='val_acc', mode='auto', restore_best_weights=True)

        

# save best model

mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='auto', save_best_only=True)



class_weights = {0: 8.80595533,

                1: 1.01597481,

                2: 0.72513282,

                3: 0.74640867,

                4: 0.84505298}





#import multiprocessing

#multiprocessing.cpu_count()



timerCallback = TimerCallback(500)



history = my_new_model.fit_generator(

    train_generator,

    steps_per_epoch = STEP_SIZE_TRAIN,

    validation_data = validation_generator, 

    validation_steps = STEP_SIZE_VALID,

    epochs = nb_epochs,

    class_weight = class_weights,

    callbacks=[mc,timerCallback,earlystopping_callback],

    workers = 2,

    use_multiprocessing = False,

    max_queue_size = 40)
import matplotlib.pyplot as plt

def plot_training(history):

    # Plot training & validation accuracy values

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



    # Plot training & validation loss values

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



    plt.savefig('acc_vs_epochs.png')

    

plot_training(history)
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.nasnet import preprocess_input, decode_predictions



saved_model = load_model('../working/best_model.h5')
import math



def calculate_score_from_predictions(df, PetID):

    mean = df[df['Filename'].str.contains(PetID)]['Predictions'].mean()

    if math.isnan(mean):

        return 4 # if no photo is available, default is Speed 4

    else: 

        return int(round(mean, 0))
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



#liefert die testdaten als iterator

test_generator = test_datagen.flow_from_directory(

    '../input/petfinder-images-331/petfinder_images_331/images_crop/real_test', # expects a single folder within

    target_size=(image_size, image_size),

    batch_size=batch_size,

    class_mode=None,

    shuffle=False)



test_generator.reset()



test_preds = saved_model.predict_generator(test_generator)

test_results=pd.DataFrame({"Filename":test_generator.filenames,

                      "Predictions":test_preds.argmax(axis=-1)})
print(test_results.shape)

test_results.Predictions.value_counts()
submit=pd.DataFrame()

submit['PetID']=test['PetID']

submit['AdoptionSpeed']=test['PetID'].apply(lambda x: calculate_score_from_predictions(test_results, x))

submit.to_csv('submission.csv',index=False)