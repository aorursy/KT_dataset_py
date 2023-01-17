# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import cv2
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import gc
import random
import re
#os.mkdir('../data/')
print(os.listdir(".."))
# Any results you write to the current directory are saved as output.

from keras import backend
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
#Total labels to classify
labels=['dog','cat']
#Helper functions to resuhffle data
#Keeping all new data in ../data directory as ../input ir RO.
def create_directory(labels):
    src_dir='../data'
    if not os.path.exists(src_dir+'/train'):
        print('Creating Directory', src_dir+'/train')
        os.makedirs(src_dir+'/train',exist_ok=True)
    else:
        print(src_dir+'/train', ' exists!')
    
        #functions to create directory with the label names
    for l in labels:
        dir_path=r''+src_dir+'/train/'+l
        if not os.path.exists(dir_path):
            print('Creating Directory',dir_path)
            os.mkdir(dir_path)
        else:
            print(dir_path, ' exists!')
                
    if not os.path.exists(src_dir+'/test/test_data'):
        print('Creating Directory', src_dir+'/test/test_data')
        os.makedirs(src_dir+'/test/test_data',exist_ok=True)
    else:
        print(src_dir+'/test/test_data', ' exists!')
        

create_directory(labels =labels)
#shutil.rmtree(path='../data')
#Function to move the images tp their corresponding folders:
def move_files(train_path,test_path):
    print('Moving Training Files ..')
    time.sleep(1)
    for i in tqdm(os.listdir(train_path)):        
        if 'dog' in i:
            shutil.copyfile(train_path+i,'../data/train/dog/'+i )
        elif 'cat' in i:
            shutil.copyfile(train_path+i,'../data/train/cat/'+i )
        else:
            print('unkown File', i)
            
    print('Moving Testing Files ..')
    time.sleep(1)
    for i in tqdm(os.listdir(test_path)):                
        shutil.copyfile(test_path+i,'../data/test/test_data/'+i )
        
    #print('File Copy in complete!')
move_files('../input/train/','../input/test/')
# Get count of number of files in this folder and all subfolders
def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])
#Setting Image and model parameters
Image_width,Image_height = 299,299
batch_size=50
Number_FC_Neurons=1024
train_dir = '../data/train/'

total_samples = get_num_files(train_dir)
val_split=0.3
n_train=total_samples*(1-val_split)
n_val=total_samples*val_split
num_classes = len(labels)
print(n_train,n_val)

gc.collect()
# Define data pre-processing 
train_image_gen = ImageDataGenerator(rescale=1/255,horizontal_flip=True,validation_split=val_split)
'''train_image_gen = ImageDataGenerator(rescale=1/255,
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        validation_split=0.3
    )'''
train_generator = train_image_gen.flow_from_directory(train_dir,target_size=(Image_width,Image_height),batch_size=batch_size,seed=42,subset='training',shuffle=True,class_mode='categorical')
val_generator = train_image_gen.flow_from_directory(train_dir,target_size=(Image_width,Image_height),batch_size=batch_size,seed=42,subset='validation',shuffle=True,class_mode='categorical')
#Load the inception model and load with its pre trained weight. But exclude the last layer aa we would train that.

InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False)    #To exclude final conv layer 
print('Inception v3 base model without last FC loaded')
#Defining the new Final conv layers 
#Using Functional APIs
x = InceptionV3_base_model.output
x_pool = GlobalAveragePooling2D()(x)
x_dense = Dense(Number_FC_Neurons,activation='relu')(x_pool)
final_pred = Dense(num_classes,activation='softmax')(x_dense)
model = Model(inputs=InceptionV3_base_model.input,outputs=final_pred)

model.summary()
#Adding Callback For early stopping.
#We use validation loss as the monitoring parameter which we need to minimize.
#Also, we'll use the result from the best epoch, thus, restore_best_weights=True(Default is False)
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

cb_checkpoint = ModelCheckpoint(filepath = '../working/best.hd5', monitor = 'val_loss', save_best_only = True, mode = 'auto',)
cb_stopping = EarlyStopping(monitor='val_loss',patience=10,mode=min,restore_best_weights=True)
cb_tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,write_images=False)
my_callback=[cb_stopping, cb_checkpoint,cb_tensorboard]
#Freeze all the layers in InceptionV3 model to train only our additional layers
'''for layer in InceptionV3_base_model.layers:
    layer.trainable=False
'''
#Fine tune model by retraining the few end layers of the inception model
layer_to_Freeze=172    
for layer in model.layers[:layer_to_Freeze]:
    layer.trainable =False
for layer in model.layers[layer_to_Freeze:]:
    layer.trainable=True
#Define model compile for basic transfer learning
#Using categorical_crossentropy loss as we need to classify only 2 classes and using softmax output layer.
#Please read difference between categorical_crossentropy and binary_crossentropy here:= https://stackoverflow.com/questions/47877083/keras-binary-crossentropy-categorical-crossentropy-confusion
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])

# Fit the transfer learning model to the data from the generators.  
# Since the data is coming from the generators that we created ,we use fit_generator() method. 
# And we need to specify the train and val_generator as the source of data.
# Please note the difference between fit() and fit_generator()

history_transfer_learning = model.fit_generator(train_generator,epochs=30,
                                                steps_per_epoch=n_train//batch_size,
                                                validation_data=val_generator,
                                                validation_steps=n_val//batch_size,
                                                verbose=1,
                                                callbacks=my_callback,
                                                class_weight='auto')

#Always save your trained model, as training process is tedious. It's better to train once and then load the already trained model.
#model.save('model.hd5')
#incase you have already trained your model, then you can load it by using following Function.
#Loading the best model.
from keras.models import load_model
print('Loading best model...')
best_model = load_model('../working/best.hd5')
print('Best model loaded!')
#os.listdir('../working')
#Clearing memory leak
#backend.clear_session()
gc.collect()
#Evaluating Metrics on Validation set
#y_pred = model.predict_generator(val_generator,verbose=1)

#Converting probability to target
#y_pred[y_pred > 0.5]=1
#y_pred[y_pred < 0.5]=0

#Use Evaluate() or evalute_generator() to check your model accuracy on validation set.
score = best_model.evaluate_generator(val_generator,verbose=1)
print('Test loss: ', score[0])
print('Test accuracy', score[1])


epoch_list = list(range(1,len(history_transfer_learning.history['acc'])+1))  #Values for x axis[1,2,3,4...# of epochs]
plt.plot(epoch_list, history_transfer_learning.history['acc'],epoch_list,history_transfer_learning.history['val_acc'])
plt.legend(('Training accuracy','Validation Accuracy'))
plt.show()


epoch_list = list(range(1,len(history_transfer_learning.history['loss'])+1))  #Values for x axis[1,2,3,4...# of epochs]
plt.plot(epoch_list, history_transfer_learning.history['loss'],epoch_list,history_transfer_learning.history['val_loss'])
plt.legend(('Training loss','Validation loss'))
plt.show()

test_dir='../data/test/'
os.listdir('../working/')
# Define data pre-processing 
test_image_gen = ImageDataGenerator(rescale=1/255)
test_generator = test_image_gen.flow_from_directory(test_dir,target_size=(Image_width,Image_height),batch_size=1,seed=42,class_mode=None,shuffle=False)
#test_generator.reset()
y_pred = model.predict_generator(generator=test_generator,verbose=1)
#submission = pd.DataFrame({'id':np.arange(1,len(test_generator.filenames)+1),'label':y_pred.clip(min=0.02,max=0.98)[:,1]})
submission = pd.DataFrame({'id':pd.Series(test_generator.filenames),'label':pd.Series(y_pred.clip(min=0.02,max=0.98)[:,1])})
#submission = pd.DataFrame({'id':pd.Series(test_generator.filenames),'label':pd.Series(y_pred[:,1])})
submission['id'] = submission.id.str.extract('(\d+)')
submission['id']=pd.to_numeric(submission['id'])
#submission.sort_values(by='id',inplace=True)

#y_pred.clip(min=0.02,max=0.98)[:100,1]
#score
#submission.nunique(axis=0)
submission.head(10)
#submission.shape
#y_pred[:15,:]
submission.to_csv('DogVsCats_submission.csv',index=False)
from keras.preprocessing.image import load_img

fig,ax = plt.subplots(5,5, figsize=(15,15))
for i,fn in enumerate(test_generator.filenames[:25]):
    path='../data/test/'+fn
    #print(path)    
    #print(i)
    img=load_img(path, target_size=(Image_width, Image_height))
    ax[i//5, i%5].imshow(img)
    ax[i//5, i%5].axis('off')
    
    if (submission['label'][i]) > 0.5:
        label='Dog'
    elif (submission['label'][i])<0.5:
            label='Cat'
    ax[i//5, i%5].set_title("It's a "+label)
plt.show()


len(model.layers)