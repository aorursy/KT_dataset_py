# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import numpy as np

import pandas as pd

import glob

import time

import pandas as pd

# from xml.dom import minidom





import os

import matplotlib.pyplot as plt

import keras



import cv2
df=pd.read_csv('/kaggle/input/cross-site-scripting-xss-dataset-for-deep-learning/XSS_dataset.csv', encoding='utf-8-sig')
df.head()
df=df[df.columns[-2:]]  # Only get sentence and labels
df.head()
# Get Sentences data from data frame

sentences=df['Sentence'].values

sentences[1]
print(len(sentences))
# Convert to ASCII



def convert_to_ascii(sentence):

    sentence_ascii=[]



    for i in sentence:

        

        

        """Some characters have values very big e.d 8221 adn some are chinese letters

        I am removing letters having values greater than 8222 and for rest greater 

        than 128 and smaller than 8222 assigning them values so they can easily be normalized"""

       

        if(ord(i)<8222):      # ” has ASCII of 8221

            

            if(ord(i)==8217): # ’  :  8217

                sentence_ascii.append(134)

            

            

            if(ord(i)==8221): # ”  :  8221

                sentence_ascii.append(129)

                

            if(ord(i)==8220): # “  :  8220

                sentence_ascii.append(130)

                

                

            if(ord(i)==8216): # ‘  :  8216

                sentence_ascii.append(131)

                

            if(ord(i)==8217): # ’  :  8217

                sentence_ascii.append(132)

            

            if(ord(i)==8211): # –  :  8211

                sentence_ascii.append(133)

                

                

            """

            If values less than 128 store them else discard them

            """

            if (ord(i)<=128):

                    sentence_ascii.append(ord(i))

    

            else:

                    pass

            



    zer=np.zeros((10000))



    for i in range(len(sentence_ascii)):

        zer[i]=sentence_ascii[i]



    zer.shape=(100, 100)





#     plt.plot(image)

#     plt.show()

    return zer





# send each sentence to be converted to ASCII





arr=np.zeros((len(sentences),100,100))



for i in range(len(sentences)):

    

    image=convert_to_ascii(sentences[i])



    x=np.asarray(image,dtype='float')

    image =  cv2.resize(x, dsize=(100,100), interpolation=cv2.INTER_CUBIC)

    image/=128



    

#     if i==1:

#         plt.plot(image)

#         plt.show()    

    arr[i]=image



print("Input data shape : ", arr.shape)
# Reshape data for input to CNN

data = arr.reshape(arr.shape[0], 100, 100, 1)
data.shape
y=df['Label'].values
# Split into train/test data

from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(data,y, test_size=0.2, random_state=42)
# import libraries for Making Model

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Activation, Conv2D, MaxPooling2D,Flatten,Dropout,MaxPool2D, BatchNormalization
# A basic CNN Model

# Number of layers = 11

# Number of Convolutional layer: 3



model=tf.keras.models.Sequential([

    

    tf.keras.layers.Conv2D(64,(3,3), activation=tf.nn.relu, input_shape=(100,100,1)),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(128,(3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Conv2D(256,(3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

    

])


model.compile(loss='binary_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])

model.summary()
# Stop when validation accuracy > 97



class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('val_accuracy')>0.97):

            print(" \n Reached 97% + validation accuracy")

            self.model.stop_training=True

callbacks = myCallback()
batch_size = 128

num_epoch = 10

#model training

model_log = model.fit(trainX, trainY,

          batch_size=batch_size,

          epochs=num_epoch,

          verbose=1,

          validation_data=( testX,  testY)

#                        callbacks=[callbacks]

                     )
# predict for test set

pred=model.predict(testX)

# Threshold values predicted



for i in range(len(pred)):

    if pred[i]>0.5:

        pred[i]=1

    elif pred[i]<=0.5:

        pred[i]=0



# Count True predicted and wrong predicted



true=0

false=0



for i in range(len(pred)):

    if pred[i] == testY[i]:

        true+=1

    else:

        false+=1

        

print("correct predicted :: ", true)

print("false prediction :: ", false)
# Number of attack and benign data in test set



attack=0

benign=0

for i in range(len(testY)):

    if testY[i]==1:

        attack+=1

    else:

        benign+=1



print("Attack data in test set :: ", attack)

print(" Benign data in test set :: ", benign)
# Takes True positive, true negative, false positive and false negative as input

# Returns Accuracy



def accuracy_function(tp,tn,fp,fn):

    

    accuracy = (tp+tn) / (tp+tn+fp+fn)

    

    return accuracy

# Takes True Positive and false positive

# Returns Precision



def precision_function(tp,fp):

    

    precision = tp / (tp+fp)

    

    return precision
#Takes True Positive and false negative

# Returns Recall



def recall_function(tp,fn):

    

    recall=tp / (tp+fn)

    

    return recall
#Takes true values and predicted

# Returns Accuracy, precision and recall





def confusion_matrix(truth,predicted):

    

    true_positive = 0

    true_negative = 0

    false_positive = 0

    false_negative = 0

    

    for true,pred in zip(truth,predicted):

        if true == 1:

            if pred == true:

                true_positive += 1

            elif pred != true:

                false_negative += 1



        elif true == 0:

            if pred == true:

                true_negative += 1

            elif pred != true:

                false_positive += 1

            

    accuracy=accuracy_function(true_positive, true_negative, false_positive, false_negative)

    precision=precision_function(true_positive, false_positive)

    recall=recall_function(true_positive, false_negative)

    

    return (accuracy,

            precision,

           recall)
accuracy,precision,recall=confusion_matrix(testY,pred)
print(" Accuracy : {0} \n Precision : {1} \n Recall : {2}".format(accuracy, precision, recall))