# Standard data science libraries

import psutil

import humanize

import os

from IPython.display import display_html



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir('/kaggle/input'))

dataDirectory= '/kaggle/input/gpds-1150/New folder (10)/'

print(os.listdir(dataDirectory))



# Any results you write to the current directory are saved as output.   
print(os.listdir('/kaggle/input'))
import numpy as np

import keras

from keras import backend as K

from keras.models import Sequential

from keras.models import Model

from keras.layers import Activation

from keras.layers.core import Dense, Flatten

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras.layers.core import Dropout

from keras.layers.convolutional import *

from keras.callbacks import ModelCheckpoint

from keras.applications.inception_v3 import InceptionV3

from keras.applications.inception_v3 import preprocess_input

from keras.applications.inception_v3 import decode_predictions

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from keras.models import model_from_json

import itertools

import matplotlib.pyplot as plt

import time

import pandas as pd

%matplotlib inline
train_path = dataDirectory+'/train/'

test_path1 = dataDirectory+'/test/1/'

test_path2 = dataDirectory+'/test/2/'

test_path3 = dataDirectory+'/test/3/'

test_path4 = dataDirectory+'/test/4/'

test_path5 = dataDirectory+'/test/5/'

test_path6 = dataDirectory+'/test/6/'

test_path7 = dataDirectory+'/test/7/'

test_path8 = dataDirectory+'/test/8/'

test_path9 = dataDirectory+'/test/9/'

test_path10 = dataDirectory+'/test/10/'

test_path11 = dataDirectory+'/test/11/'

test_path12 = dataDirectory+'/test/12/'

test_path13 = dataDirectory+'/test/13/'

test_path14 = dataDirectory+'/test/14/'

test_path15 = dataDirectory+'/test/15/'

test_path16 = dataDirectory+'/test/16/'

test_path17 = dataDirectory+'/test/17/'

test_path18 = dataDirectory+'/test/18/'

test_path19 = dataDirectory+'/test/19/'

test_path20 = dataDirectory+'/test/20/'

test_path21 = dataDirectory+'/test/21/'

test_path22 = dataDirectory+'/test/22/'

test_path23 = dataDirectory+'/test/23/'

test_path24 = dataDirectory+'/test/24/'

test_path25 = dataDirectory+'/test/25/'

test_path26 = dataDirectory+'/test/26/'

test_path27 = dataDirectory+'/test/27/'

test_path28 = dataDirectory+'/test/28/'

test_path29 = dataDirectory+'/test/29/'

test_path30 = dataDirectory+'/test/30/'

test_path31 = dataDirectory+'/test/31/'

test_path31 = dataDirectory+'/test/31/'

test_path32 = dataDirectory+'/test/32/'

test_path33 = dataDirectory+'/test/33/'

test_path34 = dataDirectory+'/test/34/'

test_path35 = dataDirectory+'/test/35/'

test_path36 = dataDirectory+'/test/36/'

test_path37 = dataDirectory+'/test/37/'

test_path38 = dataDirectory+'/test/38/'

test_path39 = dataDirectory+'/test/39/'

test_path40 = dataDirectory+'/test/40/'

test_path41 = dataDirectory+'/test/41/'

test_path42 = dataDirectory+'/test/42/'

test_path43 = dataDirectory+'/test/43/'

test_path44 = dataDirectory+'/test/44/'

test_path45 = dataDirectory+'/test/45/'

test_path46 = dataDirectory+'/test/46/'

test_path47 = dataDirectory+'/test/47/'

test_path48 = dataDirectory+'/test/48/'

test_path49 = dataDirectory+'/test/49/'

test_path50 = dataDirectory+'/test/50/'

test_path51 = dataDirectory+'/test/51/'

test_path52 = dataDirectory+'/test/52/'

test_path53 = dataDirectory+'/test/53/'

test_path54 = dataDirectory+'/test/54/'

test_path55 = dataDirectory+'/test/55/'

test_path56 = dataDirectory+'/test/56/'

test_path57 = dataDirectory+'/test/57/'

test_path58 = dataDirectory+'/test/58/'

test_path59 = dataDirectory+'/test/59/'

test_path60 = dataDirectory+'/test/60/'

test_path61 = dataDirectory+'/test/61/'

test_path62 = dataDirectory+'/test/62/'

test_path63 = dataDirectory+'/test/63/'

test_path64 = dataDirectory+'/test/64/'

test_path65 = dataDirectory+'/test/65/'

test_path66 = dataDirectory+'/test/66/'

test_path67 = dataDirectory+'/test/67/'

test_path68 = dataDirectory+'/test/68/'

test_path69 = dataDirectory+'/test/69/'

test_path70 = dataDirectory+'/test/70/'

test_path71 = dataDirectory+'/test/71/'

test_path72 = dataDirectory+'/test/72/'

test_path73 = dataDirectory+'/test/73/'

test_path74 = dataDirectory+'/test/74/'

test_path75 = dataDirectory+'/test/75/'

test_path76 = dataDirectory+'/test/76/'

test_path77 = dataDirectory+'/test/77/'

test_path78 = dataDirectory+'/test/78/'

test_path79 = dataDirectory+'/test/79/'

test_path80 = dataDirectory+'/test/80/'

test_path81 = dataDirectory+'/test/81/'

test_path82 = dataDirectory+'/test/82/'

test_path83 = dataDirectory+'/test/83/'

test_path84 = dataDirectory+'/test/84/'

test_path85 = dataDirectory+'/test/85/'

test_path86 = dataDirectory+'/test/86/'

test_path87 = dataDirectory+'/test/87/'

test_path88 = dataDirectory+'/test/88/'

test_path89 = dataDirectory+'/test/89/'

test_path90 = dataDirectory+'/test/90/'

test_path91 = dataDirectory+'/test/91/'

test_path92 = dataDirectory+'/test/92/'

test_path93 = dataDirectory+'/test/93/'

test_path94 = dataDirectory+'/test/94/'

test_path95 = dataDirectory+'/test/95/'

test_path96 = dataDirectory+'/test/96/'

test_path97 = dataDirectory+'/test/97/'

test_path98 = dataDirectory+'/test/98/'

test_path99 = dataDirectory+'/test/99/'

test_path100 = dataDirectory+'/test/100/'

test_path101 = dataDirectory+'/test/101/'

test_path102 = dataDirectory+'/test/102/'

test_path103 = dataDirectory+'/test/103/'

test_path104 = dataDirectory+'/test/104/'

test_path105 = dataDirectory+'/test/105/'

test_path106 = dataDirectory+'/test/106/'

test_path107 = dataDirectory+'/test/107/'

test_path108 = dataDirectory+'/test/108/'

test_path109 = dataDirectory+'/test/109/'

test_path110 = dataDirectory+'/test/110/'

test_path111 = dataDirectory+'/test/111/'

test_path112 = dataDirectory+'/test/112/'

test_path113 = dataDirectory+'/test/113/'

test_path114 = dataDirectory+'/test/114/'

test_path115 = dataDirectory+'/test/115/'

test_path116 = dataDirectory+'/test/116/'

test_path117 = dataDirectory+'/test/117/'

test_path118 = dataDirectory+'/test/118/'

test_path119 = dataDirectory+'/test/119/'

test_path120 = dataDirectory+'/test/120/'

test_path121 = dataDirectory+'/test/121/'

test_path122 = dataDirectory+'/test/122/'

test_path123 = dataDirectory+'/test/123/'

test_path124 = dataDirectory+'/test/124/'

test_path125 = dataDirectory+'/test/125/'

test_path126 = dataDirectory+'/test/126/'

test_path127 = dataDirectory+'/test/127/'

test_path128 = dataDirectory+'/test/128/'

test_path129 = dataDirectory+'/test/129/'

test_path130 = dataDirectory+'/test/130/'

test_path131 = dataDirectory+'/test/131/'

test_path132 = dataDirectory+'/test/132/'

test_path133 = dataDirectory+'/test/133/'

test_path134 = dataDirectory+'/test/134/'

test_path135 = dataDirectory+'/test/135/'

test_path136 = dataDirectory+'/test/136/'

test_path137 = dataDirectory+'/test/137/'

test_path138 = dataDirectory+'/test/138/'

test_path139 = dataDirectory+'/test/139/'

test_path140 = dataDirectory+'/test/140/'

test_path141 = dataDirectory+'/test/141/'

test_path142 = dataDirectory+'/test/142/'

test_path143 = dataDirectory+'/test/143/'

test_path144 = dataDirectory+'/test/144/'

test_path145 = dataDirectory+'/test/145/'

test_path146 = dataDirectory+'/test/146/'

test_path147 = dataDirectory+'/test/147/'

test_path148 = dataDirectory+'/test/148/'

test_path149 = dataDirectory+'/test/149/'

test_path150 = dataDirectory+'/test/150/'
train_datagen = ImageDataGenerator(

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        fill_mode='nearest',

    validation_split=0.2) # set validation split
selectedClasses = ['forge','genuine'] 
batchSize=32





train_generator = train_datagen.flow_from_directory(

    train_path,

    target_size=(224, 224),

    batch_size=batchSize,

    classes=selectedClasses,

    subset='training') # set as training data



validation_generator = train_datagen.flow_from_directory(

    train_path, # same directory as training data

    target_size=(224, 224),

    batch_size=batchSize,

    classes=selectedClasses,

    subset='validation') # set as validation data



test_generator = ImageDataGenerator().flow_from_directory(

    test_path150, 

    target_size=(224,224), 

    classes=selectedClasses,

    shuffle= False,

    batch_size = batchSize)# set as test data
print ("In train_generator ")

for cls in range(len (train_generator.class_indices)):

    print(selectedClasses[cls],":\t",list(train_generator.classes).count(cls))

print ("") 



print ("In validation_generator ")

for cls in range(len (validation_generator.class_indices)):

    print(selectedClasses[cls],":\t",list(validation_generator.classes).count(cls))

print ("") 



print ("In test_generator ")

for cls in range(len (test_generator.class_indices)):

    print(selectedClasses[cls],":\t",list(test_generator.classes).count(cls))
#plots images with labels within jupyter notebook

def plots(ims, figsize = (22,22), rows=4, interp=False, titles=None, maxNum = 9):

    if type(ims[0] is np.ndarray):

        ims = np.array(ims).astype(np.uint8)

        if(ims.shape[-1] != 3):

            ims = ims.transpose((0,2,3,1))

           

    f = plt.figure(figsize=figsize)

    #cols = len(ims) //rows if len(ims) % 2 == 0 else len(ims)//rows + 1

    cols = maxNum // rows if maxNum % 2 == 0 else maxNum//rows + 1

    #for i in range(len(ims)):

    for i in range(maxNum):

        sp = f.add_subplot(rows, cols, i+1)

        sp.axis('Off')

        if titles is not None:

            sp.set_title(titles[i], fontsize=20)

        plt.imshow(ims[i], interpolation = None if interp else 'none')   
train_generator.reset()

imgs, labels = train_generator.next()



#print(labels)



labelNames=[]

labelIndices=[np.where(r==1)[0][0] for r in labels]

#print(labelIndices)



for ind in labelIndices:

    for labelName,labelIndex in train_generator.class_indices.items():

        if labelIndex == ind:

            #print (labelName)

            labelNames.append(labelName)



#labels
plots(imgs, rows=4, titles = labelNames, maxNum=8)
#InceptionV3



base_model = InceptionV3(weights='imagenet', 

                                include_top=False, 

                                input_shape=(224, 224,3))

base_model.trainable = False



x = base_model.output

x = keras.layers.GlobalAveragePooling2D()(x)

# let's add a fully-connected layer

x = Dropout(0.5)(x)

predictions = Dense(len(selectedClasses), activation='sigmoid')(x)



# this is the model we will train

model = Model(input=base_model.input, output=predictions)





model.summary()
#Atutomatic rename with epoch number and val accuracy:

#filepath="checkpoints/weights-improvement-epeoch-{epoch:02d}-val_acc-{val_acc:.2f}.hdf5"





 

modelName= "Genuine_or_Forge"

#save the best weights over the same file with the model name



#filepath="checkpoints/"+modelName+"_bestweights.hdf5"

filepath=modelName+"_bestweights.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
stepsPerEpoch= (train_generator.samples+ (batchSize-1)) // batchSize

print("stepsPerEpoch: ", stepsPerEpoch)



validationSteps=(validation_generator.samples+ (batchSize-1)) // batchSize

print("validationSteps: ", validationSteps)





validationSteps=(test_generator.samples+ (batchSize-1)) // batchSize

print("validationSteps: ", validationSteps)
train_generator.reset()

validation_generator.reset()



# Fit the model

history = model.fit_generator(

    train_generator, 

    validation_data = validation_generator,

    epochs = 30,

    steps_per_epoch = stepsPerEpoch,

    validation_steps= validationSteps,

    callbacks=callbacks_list,

    verbose=1)
# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'Validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
timestr = time.strftime("%Y%m%d_%H%M%S")



# serialize model to JSON

model_json = model.to_json()

with open(timestr+"_"+modelName+"_MODEL_3"+".json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights(timestr+"_"+modelName+"_3_LAST_WEIGHTS_"+".h5")
# load json and create model

json_file = open('/kaggle/input/weights/20200106_065651_Genuine_or_Forge_MODEL_3.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

model = model_from_json(loaded_model_json)
# load weights into new model

model.load_weights("/kaggle/input/weights/20200106_065651_Genuine_or_Forge_3_LAST_WEIGHTS_.h5")
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
train_generator.reset()

score = model.evaluate_generator(train_generator, (train_generator.samples + (batchSize-1)) //batchSize)

print("For training data set of 150 users; Loss: ",score[0]," Accuracy: ", score[1])
validation_generator.reset()

score = model.evaluate_generator(validation_generator, (validation_generator.samples + (batchSize-1)) //batchSize)

print("For validation data set; Loss: ",score[0]," Accuracy: ", score[1])
test_generator.reset()

score150= model.evaluate_generator(test_generator, (test_generator.samples + (batchSize-1)) // batchSize)

print("For test data set of 1 user; Loss: ",score150[0]," Accuracy: ", score150[1])
X=plt.plot([1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150],[90.90,81.81,95.45,90.90,72.72,68.18,95.45,81.81,100,86.36,77.27,68.18,100,72.72,68.18,90.90,86.36,90.90,77.27,86.36,81.81,81.81,68.18,90.90,72.72,95.45,86.36,63.63,95.45,59.09,90.90,95.45,86.36,100,100,90.90,100,95.45,90.90,95.45,72.72,90.90,100,100,100,95.45,95.45,95.45,86.36,95.45,77.27,95.45,100,100,86.36,86.36,95.45,90.90,81.81,86.36,95.45,86.36,86.36,90.90,77.27,86.36,90.90,95.45,81.81,100,90.90,90.90,90.90,95.45,95.45,77.27,81.81,95.45,95.45,95.45,81.81,90.90,95.45,100,95.45,95.45,90.90,95.45,100,95.45,95.45,95.45,100,95.45,81.81,95.45,95.45,95.45,86.36,90.90,100,100,81.81,72.72,100,86.36,90.90,95.45,90.90,100,90.90,100,81.81,86.36,100,100,100,77.27,90.90,77.27,100,95.45,95.45,81.81,95.45,95.45,95.45,90.90,95.45,81.81,100,90.90,90.90,95.45,100,100,95.45,95.45,86.36,95.45,95.45,90.90,95.45,95.45,95.45,100,100,95.45,95.45,72.72])

plt.xlabel("Users")

plt.ylabel("Test Accuracy and loss")



Y=plt.plot([1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150],[0.3213,0.3878,0.2814,0.1610,1.8643,2.3723,0.0637,0.4960,0.0113,0.9814,0.8035,1.5303,0.0046,1.1121,1.9516,0.0944,0.5028,0.1405,0.9179,0.3127,0.8420,1.1245,1.1958,0.5847,1.6751,0.3502,0.2587,1.8981,0.0645,0.9653,0.1772,0.0441,0.3864,0.0294,0.0007,0.6175,0.01332,0.2114,0.0870,0.2223,1.5505,0.1722,0.0046,0.0025,0.0014,0.0777,0.0783,0.3116,0.5732,0.0691,0.7685,0.3665,0.0129,0.0028,0.1854,0.2531,0.1797,0.6774,0.5965,0.6225,0.0617,0.7189,0.4939,0.2869,1.0824,0.2613,0.1631,0.0854,0.5733,0.0186,0.5085,0.3300,0.1515,0.1781,0.1855,0.9256,0.3486,0.1489,0.0409,0.3419,0.6394,0.2231,0.1174,0.0090,0.4871,0.1804,0.4737,0.1496,0.0033,0.1163,0.0334,0.1304,0.0016,0.4133,0.8982,0.0640,0.0576,0.1917,0.9662,0.2296,0.0192,0.0236,0.6699,1.4994,0.0110,0.4401,0.1771,0.2491,0.3690,0.0097,0.3195,0.0097,0.4017,0.7129,0.0143,0.0077,0.0292,0.8259,0.4500,0.7601,0.0401,0.1069,0.1799,0.8583,0.1626,0.1710,0.2047,0.0958,0.1276,0.2281,0.0034,0.4001,0.2241,0.2493,0.0021,0.0363,0.1147,0.0627,0.2567,0.1146,0.1990,0.1482,0.1320,0.0712,0.0521,0.0027,0.1044,0.4130,0.0379,0.8452])

plt.xlabel("Users")

plt.ylabel("Test Accuracy and loss")
# Python program to get average of a list 

def Average(lst): 

    return sum(lst) / len(lst) 

  

# Driver Code 

lst = [90.90,81.81,95.45,90.90,72.72,68.18,95.45,81.81,100,86.36,77.27,68.18,100,72.72,68.18,90.90,86.36,90.90,77.27,86.36,81.81,81.81,68.18,90.90,72.72,95.45,86.36,63.63,95.45,59.09,90.90,95.45,86.36,100,100,90.90,100,95.45,90.90,95.45,72.72,90.90,100,100,100,95.45,95.45,95.45,86.36,95.45,77.27,95.45,100,100,86.36,86.36,95.45,90.90,81.81,86.36,95.45,86.36,86.36,90.90,77.27,86.36,90.90,95.45,81.81,100,90.90,90.90,90.90,95.45,95.45,77.27,81.81,95.45,95.45,95.45,81.81,90.90,95.45,100,95.45,95.45,90.90,95.45,100,95.45,95.45,95.45,100,95.45,81.81,95.45,95.45,95.45,86.36,90.90,100,100,81.81,72.72,100,86.36,90.90,95.45,90.90,100,90.90,100,81.81,86.36,100,100,100,77.27,90.90,77.27,100,95.45,95.45,81.81,95.45,95.45,95.45,90.90,95.45,81.81,100,90.90,90.90,95.45,100,100,95.45,95.45,86.36,95.45,95.45,90.90,95.45,95.45,95.45,100,100,95.45,95.45,72.72] 

average = Average(lst) 

  

# Printing average of the list 

print("Average of the list =", round(average, 150)) 

# Python program to get average of a list 

def Average(lst): 

    return sum(lst) / len(lst) 

  

# Driver Code 

lst=[0.3213,0.3878,0.2814,0.1610,1.8643,2.3723,0.0637,0.4960,0.0113,0.9814,0.8035,1.5303,0.0046,1.1121,1.9516,0.0944,0.5028,0.1405,0.9179,0.3127,0.8420,1.1245,1.1958,0.5847,1.6751,0.3502,0.2587,1.8981,0.0645,0.9653,0.1772,0.0441,0.3864,0.0294,0.0007,0.6175,0.01332,0.2114,0.0870,0.2223,1.5505,0.1722,0.0046,0.0025,0.0014,0.0777,0.0783,0.3116,0.5732,0.0691,0.7685,0.3665,0.0129,0.0028,0.1854,0.2531,0.1797,0.6774,0.5965,0.6225,0.0617,0.7189,0.4939,0.2869,1.0824,0.2613,0.1631,0.0854,0.5733,0.0186,0.5085,0.3300,0.1515,0.1781,0.1855,0.9256,0.3486,0.1489,0.0409,0.3419,0.6394,0.2231,0.1174,0.0090,0.4871,0.1804,0.4737,0.1496,0.0033,0.1163,0.0334,0.1304,0.0016,0.4133,0.8982,0.0640,0.0576,0.1917,0.9662,0.2296,0.0192,0.0236,0.6699,1.4994,0.0110,0.4401,0.1771,0.2491,0.3690,0.0097,0.3195,0.0097,0.4017,0.7129,0.0143,0.0077,0.0292,0.8259,0.4500,0.7601,0.0401,0.1069,0.1799,0.8583,0.1626,0.1710,0.2047,0.0958,0.1276,0.2281,0.0034,0.4001,0.2241,0.2493,0.0021,0.0363,0.1147,0.0627,0.2567,0.1146,0.1990,0.1482,0.1320,0.0712,0.0521,0.0027,0.1044,0.4130,0.0379,0.8452]

average = Average(lst) 

  

# Printing average of the list 

print("Average of the list =", round(average, 150)) 

lst = [0.3213,0.3878,0.2814,0.1610,0.1610,1.8643,2.3723,0.0637,0.4960,0.0113,0.9814,0.9814,0.8035,1.5303,0.0046,1.1121,1.9516,0.0944,0.5028,0.1405,0.9179,0.3127,0.8420,1.1245,1.1958,0.5847,1.6751,0.3502,0.2587,1.8981,0.0645,0.9653,0.1772,0.0441,0.3864,0.0294,0.0007,0.6175,0.01332,0.2114,0.0870,0.223,1.5505,0.1722,0.0046,0.0025,0.0014,0.0777,0.0783,0.3116,0.5732,0.0691,0.7685,0.3665,0.0129,0.0028,0.1854,0.2531,0.1797,0.6774,0.5965,0.6225,0.0617,0.7189,0.4939,0.2869,1.0824,0.2613,0.1631,0.0854,0.5733,0.0186,0.5085,0.3300,0.1515,0.1781,0.1855,0.9256,0.3486,0.1489,0.0409,0.3419,0.6394,0.2231,0.1174,0.0090,0.4871,0.1804,0.4737,0.1496,0.0033,0.1163,0.0334,0.1304,0.0016,0.4133,0.8982,0.0640,0.0576,0.1917,0.9662,0.2296,0.0192,0.0236,0.6699,1.4994,0.0110,0.4401,0.1771,0.2491,0.3690,0.0097,0.3195,0.0097,0.4017,0.7129,0.0143,0.0077,0.0292,0.8259,0.4500,0.7601,0.0401,0.1069,0.1799,0.8583,0.1626,0.1710,0.2047,0.0958,0.1276,0.2281,0.0034,0.4001,0.2241,0.2241,0.2493,0.0021,0.0363,0.1147,0.0627,0.2567,0.1146,0.1990,0.1482,0.1320,0.0712,0.0521,0.0027,0.1044,0.4130,0.0379,0.8452] 

max_value=max(lst)

print(max_value)
test_generator.reset()

testStep = (test_generator.samples + (batchSize-1)) // batchSize

print("testStep: ", testStep)

predictions = model.predict_generator(test_generator, steps = testStep ,  verbose = 1)

len(predictions)
len(predictions)
predicted_class_indices=np.argmax(predictions,axis=1)

print(predicted_class_indices)

len(predicted_class_indices)
labels = (test_generator.class_indices)

print(labels)
labels = dict((v,k) for k,v in labels.items())

print(labels)
predictedLables= [labels[k] for k in predicted_class_indices]

print(predictedLables)

len(predictedLables)
actualLables= [labels[k] for k in test_generator.classes]

print(actualLables)

len(actualLables)
accuracy_score(actualLables, predictedLables)
matrix = confusion_matrix(actualLables, predictedLables)

print(labels)

matrix
print(classification_report(actualLables, predictedLables))
recall_score( actualLables, predictedLables,average='weighted') 
precision_score( actualLables, predictedLables,average='weighted') 
#Prepared code that is taken from SKLearn Website, Creates Confusion Matrix

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cm_plot_labels = selectedClasses

plot_confusion_matrix(matrix,cm_plot_labels, normalize=False

                      , title = 'Confusion Matrix')
filenames=test_generator.filenames

directory= test_generator.directory

results=pd.DataFrame({"Directory":directory,

                      "Filename":filenames,

                      "Predictions":predictedLables,

                     "Actuals": actualLables })

results.to_csv("results30.csv",index=False)
#import glob

#import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



res = results[1:22]



images = []

#for img_path in glob.glob('images/*.jpg'):

for img_path in res['Directory']+"/"+res['Filename']:

    images.append(mpimg.imread(img_path))



plt.figure(figsize=(80,80))

columns = 4

for i, image in enumerate(images):

    ax= plt.subplot(len(images) / columns + 1, columns, i + 1)

    ax.set_title(res['Actuals'].iloc[i]+" "+res['Predictions'].iloc[i], fontsize=40)

    plt.imshow(image)