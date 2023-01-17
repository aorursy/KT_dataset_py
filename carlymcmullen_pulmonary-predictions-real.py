#begin by importing programs 

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

%matplotlib inline

from PIL import Image 

import glob

from pathlib import Path

import seaborn as sns

import tensorflow as tf



tf.random.set_seed(16)

np.random.seed(11)
print(os.listdir("../input"))
#explore main directory

mainDIR = os.listdir('../input/chest-xray-pneumonia/chest_xray/chest_xray')

print(mainDIR)
#create paths to each individual folder 

train_folder = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray/train')

val_folder = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray/val')

test_folder = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
#train data pathways

normal_cases_dir = train_folder / 'NORMAL'

pneu_cases_dir = train_folder / 'PNEUMONIA'



normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneu_cases_dir.glob('*.jpeg')



train_data = []



#append data to list

for img in normal_cases:

    train_data.append((img,0))

    

for img in pneumonia_cases:

    train_data.append((img, 1))



#convert to pandas dataframe

train_data = pd.DataFrame(train_data, columns=['image', 'label'], index=None)



#shuffle

train_data = train_data.sample(frac=1.).reset_index(drop=True)


print(train_data.shape)

train_data.head()
#validation data pathways

normal_cases_dir = val_folder / 'NORMAL'

pneu_cases_dir = val_folder / 'PNEUMONIA'



normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneu_cases_dir.glob('*.jpeg')



val_data = []



#append data to list

for img in normal_cases:

    val_data.append((img,0))

    

for img in pneumonia_cases:

    val_data.append((img, 1))



#convert to pandas dataframe

val_data = pd.DataFrame(val_data, columns=['image', 'label'], index=None)



#shuffle

val_data = val_data.sample(frac=1.).reset_index(drop=True)
print(val_data.shape)

val_data.head()
#test data pathways

normal_cases_tdir = test_folder / 'NORMAL'

pneu_cases_tdir = test_folder / 'PNEUMONIA'



normal_tcases = normal_cases_tdir.glob('*.jpeg')

pneumonia_tcases = pneu_cases_tdir.glob('*.jpeg')



test_data = []



#append data to list

for img in normal_tcases:

    test_data.append((img,0))

    

for img in pneumonia_tcases:

    test_data.append((img, 1))



#convert to pandas dataframe

test_data = pd.DataFrame(test_data, columns=['image', 'label'], index=None)



#shuffle

test_data = test_data.sample(frac=1.).reset_index(drop=True)
print(test_data.shape)

test_data.head()
#how many cases each? 



f = plt.figure(figsize=(10,6))



a1 = f.add_subplot(1,2,1)

sns.countplot(train_data.label, ax = a1)

a1.set_title('Train Data')

a1.set_xticklabels(['Normal', 'Pneumonia'])



a2 = f.add_subplot(1,2,2)

sns.countplot(test_data.label, ax=a2)

a2.set_title('Test Data')

a2.set_xticklabels(['Normal', 'Pneumonia'])
f = plt.figure(figsize=(12,8))



a1 = f.add_subplot(1,2,1)

img_plot = plt.imshow(Image.open(train_data.image[0]))

a1.set_title('Pneumonia')



a2 = f.add_subplot(1,2,2)

img_plot = plt.imshow(Image.open(train_data.image[2]))

a2.set_title('Normal')
#make a list with each image size 

shapes = []



for x in range(0,5215):

    shapes.append(plt.imread(train_data.image[x]).shape)
#find the smallest one

min(shapes)
#importing necessary programs 

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from PIL import Image 

from scipy import ndimage 
#scale images

train_dir = ImageDataGenerator(rescale=1./255)

val_dir = ImageDataGenerator(rescale=1./255)

test_dir = ImageDataGenerator(rescale=1./255)
#Make Directories 

train_gen = train_dir.flow_from_directory(train_folder, target_size=(127,384), batch_size = 20, class_mode='binary')

val_gen = val_dir.flow_from_directory(val_folder, target_size=(127,384), batch_size=10, class_mode='binary')

test_gen = test_dir.flow_from_directory(test_folder, target_size=(127,384), batch_size=20, class_mode='binary')
from keras import layers 

from keras import models 

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from keras.models import Sequential 



# CNN model 



model_1 = Sequential()



#Convolution

model_1.add(Conv2D(32, (3,3), activation ='relu', input_shape=(127, 384, 3)))



#Pooling

model_1.add(MaxPooling2D(pool_size=(2,2)))



#Convolution 2 

model_1.add(Conv2D(64, (3,3), activation='relu'))



#Pooling Layer 2 

model_1.add(MaxPooling2D(pool_size=(2,2)))



#Convolution 3

model_1.add(Conv2D(128, (3,3), activation='relu'))



#Pooling again

model_1.add(MaxPooling2D(pool_size=(2,2)))



#Convolution 4

model_1.add(Conv2D(128, (3,3), activation='relu'))



#Flattening

model_1.add(Flatten())



#Add Dense Layers 

model_1.add(Dense(512, activation='relu'))

model_1.add(Dense(1, activation='sigmoid'))
#Compile



from keras import optimizers 



model_1.compile(loss='binary_crossentropy',

           optimizer=optimizers.RMSprop(lr=1e-4),

           metrics=['acc'])
history = model_1.fit_generator(train_gen,

                           steps_per_epoch=100,

                           epochs=20,

                           validation_data=val_gen,

                           validation_steps=50)
test_1_acc = model_1.evaluate_generator(test_gen, steps=50)

print('The accuracy of this test is:', test_1_acc[1]*100,'%')
from keras.layers import Dropout



model_2 = Sequential()



#Convolution

model_2.add(Conv2D(32, (3,3), activation ='relu', input_shape=(127, 384, 3)))



#Pooling

model_2.add(MaxPooling2D(pool_size=(2,2)))



#Convolution 2 

model_2.add(Conv2D(64, (3,3), activation='relu', padding='same'))



#Dropout 1

model_2.add(Dropout(0.35))



#Pooling Layer 2 

model_2.add(MaxPooling2D(pool_size=(2,2)))



#Convolution 3

model_2.add(Conv2D(128, (3,3), activation='relu', padding='same'))



#Dropout 2

model_2.add(Dropout(0.35))



#Pooling again

model_2.add(MaxPooling2D(pool_size=(2,2)))



#Convolution 4

model_2.add(Conv2D(256, (3,3), activation='relu', padding='same'))



#Dropout 3

model_2.add(Dropout(0.35))



#Flattening

model_2.add(Flatten())



#Add Dense Layers 

model_2.add(Dense(512, activation='relu'))

model_2.add(Dense(128, activation='relu'))

model_2.add(Dense(1, activation='sigmoid'))
model_2.compile(loss='binary_crossentropy',

           optimizer=optimizers.RMSprop(lr=1e-4),

           metrics=['acc'])
history_2 = model_2.fit_generator(train_gen,

                           steps_per_epoch=100,

                           epochs=30,

                           validation_data=val_gen,

                           validation_steps=50)
test_4_acc = model_2.evaluate_generator(test_gen, steps=50)

print('The accuracy of this test is:', test_4_acc[1]*100,'%')
train_data.label.value_counts()
pneu_count = train_data.label.value_counts(1)[1]

norm_count = train_data.label.value_counts(1)[0]

weight_for_0 = (1/norm_count)

weight_for_1 = (1/pneu_count)



class_weight = {0: weight_for_0, 1: weight_for_1}
history = model_2.fit_generator(train_gen,

                                steps_per_epoch=100,

                                epochs=30,

                                validation_data=val_gen,

                                validation_steps=50,

                               class_weight = class_weight)
test_4_acc = model_2.evaluate_generator(test_gen, steps=50)

print('The accuracy of this test is:', test_4_acc[1]*100,'%')
#import network and create CNN base

from keras.applications import VGG19

cnn_base = VGG19(weights='imagenet',

                include_top = False,

                input_shape=(127,384,3))
#build the dense layer of the model w/ cnn_base as convolutional model 

model_pt = models.Sequential()

model_pt.add(cnn_base)

model_pt.add(layers.Flatten())

model_pt.add(layers.Dense(132, activation='relu'))

model_pt.add(layers.Dense(1, activation='sigmoid'))
#checking trainable layers and trainable weights 

for layer in model_pt.layers:

    print(layer.name, layer.trainable)

    

print(len(model_pt.trainable_weights))
#we will now "freeze" the layer by setting the trainable attribute to false



cnn_base.trainable = False 
#repeat this for a sanity check 

for layer in model_pt.layers:

    print(layer.name, layer.trainable)

    

print(len(model_pt.trainable_weights))
#compile model 

METRICS = [

        'accuracy',

        tf.keras.metrics.Precision(name='precision'),

        tf.keras.metrics.Recall(name='recall')

    ]



model_pt.compile(loss='binary_crossentropy',

             optimizer=optimizers.RMSprop(lr=2e-5),

             metrics= METRICS)
#fit



history = model_pt.fit_generator(train_gen,

                               steps_per_epoch=27,

                               epochs=7,

                               validation_data=val_gen,

                               validation_steps=10,

                             class_weight=class_weight)
#visualizing training accruacy and loss 

fig, ax = plt.subplots(1, 4, figsize=(20, 3))

ax = ax.ravel()



for i, met in enumerate(['precision', 'recall', 'accuracy', 'loss']):

    ax[i].plot(history.history[met])

    ax[i].plot(history.history['val_' + met])

    ax[i].set_title('Model {}'.format(met))

    ax[i].set_xlabel('epochs')

    ax[i].set_ylabel(met)

    ax[i].legend(['train', 'val'])
loss, acc, prec, rec =model_pt.evaluate(test_gen)
#first, unfreeze the base.



cnn_base.trainable = True
#iterate thru the layers 

set_trainable = False

for layer in cnn_base.layers:

    if layer.name == 'block5_conv1':

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False 
#remodel 

model_ft = models.Sequential()

model_ft.add(cnn_base)

model_ft.add(layers.Flatten())

model_ft.add(layers.Dense(132, activation='relu'))

model_ft.add(layers.Dense(1, activation='sigmoid'))
model_ft.compile(loss='binary_crossentropy',

             optimizer = optimizers.RMSprop(lr=1e-4),

             metrics=METRICS)
batch_size = 50



history = model_ft.fit_generator(train_gen,

                               steps_per_epoch=27,

                               epochs=7,

                               validation_data=val_gen,

                               validation_steps=10,

                             class_weight=class_weight)
loss, acc, prec, rec =model_ft.evaluate(test_gen)
#visualizing training accruacy and loss 

fig, ax = plt.subplots(1, 4, figsize=(20, 3))

ax = ax.ravel()



for i, met in enumerate(['precision', 'recall', 'accuracy', 'loss']):

    ax[i].plot(history.history[met])

    ax[i].plot(history.history['val_' + met])

    ax[i].set_title('Model {}'.format(met))

    ax[i].set_xlabel('epochs')

    ax[i].set_ylabel(met)

    ax[i].legend(['train', 'val'])
from sklearn.metrics import accuracy_score, confusion_matrix

loss, acc, prec, rec =model_ft.evaluate(test_gen)
import sklearn.metrics as metrics



test_gen_final = test_dir.flow_from_directory(test_folder, target_size=(127,384),batch_size=624, class_mode='binary')

test_data, test_labels = next(test_gen_final)

preds = model_ft.predict(test_data)



fpr, tpr, threshold = metrics.roc_curve(test_labels, np.round(preds))

roc_auc = metrics.auc(fpr, tpr)



import matplotlib.pyplot as plt

plt.title('ROC Curve')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
acc = accuracy_score(test_labels, np.round(preds))*100

cm = confusion_matrix(test_labels, np.round(preds))

tn, fp, fn, tp = cm.ravel()



#confusion matrix

from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm ,  figsize=(5, 5))

plt.show()



print('\nTEST METRICS ----------------------')

precision = tp/(tp+fp)*100

recall = tp/(tp+fn)*100

sensitivity = recall

specificity = tn/(tn+fp)*100

print('Accuracy: {}%'.format(acc))

print('Precision: {}%'.format(precision))

print('Recall: {}%'.format(recall))

print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

print('Specificity: {}%'.format(specificity))

print('Sensitivity: {}%'.format(recall))