import random # create random numbers 
import collections # dictionaries in programming
random.seed(7)

import tensorflow as tf # Deep learning libraries
# These are for run the model: Convolutional Neural Networks
from tensorflow.keras import layers,models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import SensitivityAtSpecificity

#this is for plotting
import matplotlib.pyplot as plt
from time import time #calculate time
import tensorflow.keras.backend as k
import tensorflow.keras.callbacks as Callback

import numpy as np # linear algebra libraries 
import pandas as pd # data processing, to read data CSV file I/O (e.g. pd.read_csv)

import os # loading stuff
#sklearn
from sklearn.metrics import classification_report, confusion_matrix #very important in medicine

# metrics to check if the model is predicting correctly
from sklearn.model_selection import cross_val_score # cross-validation
from sklearn.metrics import roc_auc_score # Area under the curve AUC
from sklearn.metrics import roc_curve  # ROC curve
# In case we have GPU e.g. Kaggle notebook we use it, otherwise we used CPU
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found in this computer")
else:
    print("Found GPU at: {}".format(device_name))

pwd
#The data was read from three directories: train, validation and test # kaggle
train_dir = '/kaggle/input/train' #'/kaggle/input/pneumonia/pneumonia2/train'
val_dir = '/kaggle/input/validation'#'/kaggle/input/pneumonia/pneumonia2/validation'
test_dir = '/kaggle/input/test'#'/kaggle/input/pneumonia/pneumonia2/test'
os.listdir(test_dir)
folders = ['train', 'validation', 'test']
for i in folders:
    print(i)
    DL_ = os.listdir('/kaggle/input/{}/DL'.format(i)) # dir is your directory path
    RDL_ = os.listdir('/kaggle/input/{}/RDL'.format(i)) 
    print(len(DL_), len(RDL_), ", % of DL / total =", 100*(len(DL_)/(len(RDL_) + len(DL_))))
# Because there wasn't enough data, I augmentated the data: pseudoreplication of data
# photos reused in funky angles
val_datagen = ImageDataGenerator(rescale=1./255)
test_generator = val_datagen.flow_from_directory(
        test_dir,
        #verbose = 0,
        shuffle=False,
        target_size = (150, 150),
        batch_size = 1, #<----tensorflow documentation: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
        class_mode='binary')
# Actual Model:
def model_(neurones, batch): #specify the number of neurons, batch size
    
    #getting DIFFERENT metrics for Accuracy 
    Preci = tf.keras.metrics.Precision() 
    Recal = tf.keras.metrics.Recall() 
    
    #Image generator to rescale images: rotate randomly about 5 degrees the photo, and flip the photo
    train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range = 5, 
                                    horizontal_flip = True) # here we could use more features for augmentation
    #it was not necessary
    
    # Photo preparation
    val_datagen = ImageDataGenerator(rescale=1./255) # validation set: used black and white

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (150, 150), # resize images
        batch_size = batch,
        class_mode = 'binary')
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size = (150, 150),
        batch_size = batch,
        class_mode='binary')
    
    try:
        with tf.device('/device:GPU:0'): #In case we have GPU use it:
            #9 layers convolutional NN, with RELU activation (wikipedia), the last layer used sigmoid activation
            model = tf.keras.models.Sequential()
            model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation = 'relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation = 'relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation = 'relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation = 'relu'))
            model.add(MaxPooling2D((2, 2)))            
            model.add(Conv2D(32, (3, 3), activation = 'relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(16, (3, 3), activation = 'relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Flatten()) #you make it one dimension
            model.add(Dense(neurones, activation = 'relu'))
            model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.05)))
            model.add(Dense(1, activation = 'sigmoid')) # last layer sigmoid activation because
            #you have binary classification problem, DL vs RDL

            model.compile(loss = 'binary_crossentropy',
                          optimizer = tf.keras.optimizers.RMSprop(lr = 1e-4),
                          metrics = ['acc', Preci, Recal])
            print("GPU/TPU IS ON")
    except: #when we dont have a GPU:
        
            model = tf.keras.models.Sequential()
            #model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
            #model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation = 'relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(32, (3, 3), activation = 'relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(16, (3, 3), activation = 'relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.Flatten())
            model.add(Dense(neurones, activation = 'elu'))
            model.add(Dense(256, activation='elu',kernel_regularizer=regularizers.l1(0.01)))
            model.add(Dense(1, activation = 'sigmoid'))

            model.compile(loss = 'binary_crossentropy',
                          optimizer = tf.keras.optimizers.RMSprop(lr = 1e-4),
                          metrics = ['acc', Preci, Recal])
            print("NOT USING GPU")

    return model, train_generator, val_generator
%%capture
# Here some hyper-parameters that were used:

#1. Number neurones in the dense layer:
neurones = [512] # This was implemented earlier, and perhaps
#useful in the future for us, we leave it here (sorry!)

batches = [32] # List for batch size evaluation requested by S. Lukkarinen

# Dictionaries: e.g. Models{ model_NN_Batch_size, model_NN_Batch_size, and model_NN_batch_size}
models = {}
train_generators = {}
val_generators = {}

for ss in neurones: 
    for batch in batches:
        model_ss_batch = 'models_{}_{}'.format(ss, batch)  # creating the keys
        models[model_ss_batch] = model_(ss, batch) # appending the model to the key
        
        train_generator_ss_batch = 'train_generators_{}_{}'.format(ss, batch)  # creating the keys
        train_generators[train_generator_ss_batch] = model_(ss, batch) # appending the generator to the key
        
        val_generator_ss_batch = 'val_generators_{}_{}'.format(ss, batch)  # creating the keys
        val_generators[val_generator_ss_batch] = model_(ss, batch) # appending the model to the key       
        
# Only to check the correct appending in the dictionary 
print(models.keys()) #prints keys
#print(train_generators.keys()) #prints keys
##print(val_generators.keys()) #prints keys
# Model Fitting step (training)

Final_models = {} # create a dictionary for FITTED models
for i in range(len(models)):
    print('MODEL:',i)
    Final_model_i = 'models_{}'.format(i)  # creating the keys
    
    # and here the fitting, and the model will be appended to the dictionary
    Final_models[Final_model_i] = models[list(models.keys())[i]][0].fit_generator(
        train_generators[list(train_generators.keys())[i]][1],
        steps_per_epoch = None, 
        verbose = 2,
        epochs = 300, # we used 30 since the process was very very slow..and metrics tend to stabilize at 10 epochs
        validation_data = val_generators[list(val_generators.keys())[i]][1],
        validation_steps = None, 
    )
    # saving the model
    #models[list(models.keys())[i]][0].save("model_mar{}.h5".format(i)) # save three models for evaluation

models[list(models.keys())[0]][0].save("model_mar{}.h5".format(0)) 
# STEP: Checking the performance of the model:
class EVALUATE:
    
    def fitting_evaluation():

        for i in range(len(Final_models)):
            print(i)
            models_key = Final_models.keys()
            model_key = [j  for  j in  models_key]
            metrics = Final_models[list(Final_models.keys())[i]].history.keys()
            metric = [k  for  k in  metrics]

            # Accuracy
            acc = Final_models[list(Final_models.keys())[i]].history[metric[1]]
            val_acc = Final_models[list(Final_models.keys())[i]].history[metric[5]]
            #Precision
            preci = Final_models[list(Final_models.keys())[i]].history[metric[2]]
            val_preci = Final_models[list(Final_models.keys())[i]].history[metric[6]]
            #Recall
            recal = Final_models[list(Final_models.keys())[i]].history[metric[3]]
            val_recal = Final_models[list(Final_models.keys())[i]].history[metric[7]]
            #Loss 
            loss = Final_models[list(Final_models.keys())[i]].history[metric[0]]
            val_loss = Final_models[list(Final_models.keys())[i]].history[metric[4]]

            epochs = range(len(acc))

            fig, axs = plt.subplots(1, 4, figsize=(10, 9))


            axs[ 0].set_title('1. Accuracy')
            axs[ 0].plot(epochs, acc, 'bo-', label = 'Training acc')
            axs[ 0].plot(epochs, val_acc, 'r*-', label = 'Validation acc')

            axs[ 1].set_title('2. Precision ')
            axs[ 1].plot(epochs, preci, 'bo-', label = 'Precision_training ')
            axs[ 1].plot(epochs, val_preci, 'r*-', label = 'Precision validation')

            axs[ 2].set_title('3. Recall (sensitivity) ')
            axs[ 2].plot(epochs, recal, 'bo-', label = 'Recall training')
            axs[ 2].plot(epochs, val_recal, 'r*-', label = 'Recall validation')

            axs[ 3].set_title('4. Loss')
            axs[ 3].plot(epochs, loss, 'bo-', label = 'Training loss')
            axs[ 3].plot(epochs, val_loss, 'r*-', label = 'Validation loss')


            fig.suptitle(model_key[i], fontname="Times New Roman",fontweight="bold")
            fig.text(0.5, 0.04, 'EPOCH', ha='center', fontname="Times New Roman",fontweight="bold")
            plt.show()
    

    def metric_evaluation():
        test_generator.reset() # re-setting generator
        for i in range(0, len(Final_models)):
            Loss, Accuracy, Preci, Recal = models[list(models.keys())[i]][0].evaluate_generator(generator=test_generator)
            print("Model", i)
            print('Loss: {}'.format(Loss), 'Accuracy: {}'.format(Accuracy), 'Precision: {}'.format(Preci), 'Recall: {}'.format(Recal))

    
    
    def confusion_threshold_evaluation(cutoff): 

        Y_labels = test_generator.classes

        for i in range(0, len(Final_models)):
            print("Prediction for model {}:".format(i))

            Y_pred = models[list(models.keys())[i]][0].predict_generator(test_generator)
            Y_pred_prob = Y_pred
            Y_pred = 1*(Y_pred.astype('float64') > cutoff)
            

            print('Confusion Matrix')
            print(confusion_matrix(Y_labels, Y_pred))
            print('Classification Report')
            target_names = ['1','0']
            print(classification_report(Y_labels, Y_pred, target_names=target_names))

            logit_roc_auc = roc_auc_score(Y_labels, Y_pred)

            fpr, tpr, thresholds = roc_curve(Y_labels, Y_pred_prob)


            fig, axs = plt.subplots(1, 2, figsize=(10, 10))

            axs[0].set_title('1. ROC')
            axs[0].plot(fpr, tpr, label='MODEL 1 (area = %0.2f)' % logit_roc_auc)
            axs[0].plot([0, 1], [0, 1],'r--')
            axs[0].set_xlabel('False Positive Rate')
            axs[0].set_ylabel('True Positive Rate')
            axs[0].legend(loc="lower right")

            axs[1].set_title('2. Threshold analysis ')
            axs[1].plot(thresholds, 1 - fpr, label = 'Specificity')
            axs[1].plot(thresholds, tpr, 'r*-', label = 'Sensitivity')
            axs[1].axvline(cutoff, color = 'black', linestyle = ":")
            axs[1].set_xlim([0, 1])
            axs[1].legend(loc="lower right")
            axs[1].set_xlabel('Threshold')
            axs[1].set_ylabel('Metrics value')

            plt.show()        
        
fit_evaluation= EVALUATE.fitting_evaluation()
# Checking against test data, which has never been seen by the model
metric_evaluation = EVALUATE.metric_evaluation()
EVALUATE.confusion_threshold_evaluation(0.5)
EVALUATE.confusion_threshold_evaluation(0.1)
EVALUATE.confusion_threshold_evaluation(0.8)
EVALUATE.confusion_threshold_evaluation(0.85)
EVALUATE.confusion_threshold_evaluation(0.9)
#threshold=0.5
print((40+29)/(40+29+10+22))
#threshold 0.1
print((66)/(66+35))
#threshold:
print((72)/(72+29))
#threshold:
print((72)/(72+29))
#threshold 0.85
print((73)/(73+28))
#threshold 0.9
print((72)/(72+29))
# 21 % improvement