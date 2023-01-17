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
import tensorflow as tf

print(tf.__version__)
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
#from https://www.kaggle.com/christofhenkel/weighted-kappa-loss-for-keras-tensorflow



import tensorflow as tf



def kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=5, bsize=32, name='kappa_loss'):

    """A continuous differentiable approximation of discrete kappa loss.

        Args:

            y_pred: 2D tensor or array, [batch_size, num_classes]

            y_true: 2D tensor or array,[batch_size, num_classes]

            y_pow: int,  e.g. y_pow=2

            N: typically num_classes of the model

            bsize: batch_size of the training or validation ops

            eps: a float, prevents divide by zero

            name: Optional scope/name for op_scope.

        Returns:

            A tensor with the kappa loss."""



    with tf.name_scope(name):

        y_true = tf.to_float(y_true)

        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))

        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))

        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)

    

        pred_ = y_pred ** y_pow

        try:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))

        except Exception:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

    

        hist_rater_a = tf.reduce_sum(pred_norm, 0)

        hist_rater_b = tf.reduce_sum(y_true, 0)

    

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

    

        nom = tf.reduce_sum(weights * conf_mat)

        denom = tf.reduce_sum(weights * tf.matmul(

            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /

                              tf.to_float(bsize))

    

        return nom / (denom + eps)
def kappa_metric(y_pred, y_true, y_pow=2, eps=1e-10, N=5, bsize=32, name='kappa_metric'):

    """A continuous differentiable approximation of discrete kappa loss.

        Args:

            y_pred: 2D tensor or array, [batch_size, num_classes]

            y_true: 2D tensor or array,[batch_size, num_classes]

            y_pow: int,  e.g. y_pow=2

            N: typically num_classes of the model

            bsize: batch_size of the training or validation ops

            eps: a float, prevents divide by zero

            name: Optional scope/name for op_scope.

        Returns:

            A tensor with the kappa loss."""



    with tf.name_scope(name):

        y_true = tf.to_float(y_true)

        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))

        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))

        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)

    

        pred_ = y_pred ** y_pow

        try:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))

        except Exception:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

    

        hist_rater_a = tf.reduce_sum(pred_norm, 0)

        hist_rater_b = tf.reduce_sum(y_true, 0)

    

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

    

        nom = tf.reduce_sum(weights * conf_mat)

        denom = tf.reduce_sum(weights * tf.matmul(

            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /

                              tf.to_float(bsize))

    

        return 1 - (nom / (denom + eps))
from tensorflow.keras.applications import NASNetLarge

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

from tensorflow.keras.optimizers import Adam, Nadam



num_classes = 5

#try pooling max

model = NASNetLarge(weights='imagenet', include_top=False, pooling='avg')



my_new_model = Sequential()

my_new_model.add(model)

my_new_model.add(Dense(512, activation="relu"))

my_new_model.add(Dropout(rate=0.30))

my_new_model.add(Dense(512, activation="relu"))

my_new_model.add(Dropout(rate=0.30))                 

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# descent optimizer (adam lr defaut = 0.001)

my_new_model.compile(optimizer=Adam(), loss=kappa_loss, metrics=[kappa_metric])



print(my_new_model.summary())
layers = [(layer, layer.name, layer.trainable) for layer in my_new_model.layers]

pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])   
from tensorflow.keras.applications.nasnet import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.callbacks import Callback



image_size = 331

batch_size = 16

nb_epochs = 50



# steps_per_epoch: number of yields (batches) before a epoch is over

# ceil(num_samples / batch_size)

# epochs: Number of epochs to train the model. An epoch is an iteration over the entire data provided

# class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). 

#  This can be useful to tell the model to "pay more attention" to samples from an under-represented class.

shift = 0.2



#brightness_range=[0.8,1.2],



train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   zoom_range=[0.8,1.2],width_shift_range=shift, height_shift_range=shift, 

                                   horizontal_flip=True)



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

earlystopping_callback = EarlyStopping(patience=batch_size/10, monitor='val_loss', mode='auto', restore_best_weights=True)

        

# save best model

mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', save_best_only=True)



class_weights = {0: 8.94366197,

                1: 1.03206265,

                2: 0.70208061,

                3: 0.73782613,

                4: 0.87751256}



timerCallback = TimerCallback(500)



history = my_new_model.fit_generator(

    train_generator,

    steps_per_epoch = STEP_SIZE_TRAIN,

    validation_data = validation_generator, 

    validation_steps = STEP_SIZE_VALID,

    epochs = nb_epochs,

    class_weight = None,

    callbacks=[tensorboard_callback, mc,timerCallback],

    workers = 2,

    use_multiprocessing = False,

    max_queue_size = 40)
import matplotlib.pyplot as plt

def plot_training(history):

    # Plot training & validation accuracy values

    plt.plot(history.history['kappa_metric'])

    plt.plot(history.history['val_kappa_metric'])

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



saved_model = load_model('../working/best_model.h5', custom_objects={'kappa_loss': kappa_loss, 'kappa_metric': kappa_metric})
import math



def calculate_score_from_predictions(df, PetID):

    mean = df[df['Filename'].str.contains(PetID)]['Predictions'].mean()

    if math.isnan(mean):

        return 4 # if no photo is available, default is Speed 4

    else: 

        return int(round(mean, 0))
test = pd.read_csv('../input/petfinder-images/petfinder_images/images/real_test/test.csv')
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



#liefert die testdaten als iterator

test_generator = test_datagen.flow_from_directory(

    '../input/petfinder-images/petfinder_images/images/real_test', # expects a single folder within

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