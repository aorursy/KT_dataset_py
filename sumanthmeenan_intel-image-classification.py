#Import libraries

import numpy as np

import pandas as pd 

import os

import glob

import random

import cv2

import matplotlib.pyplot as plt

import shutil



import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

from keras.callbacks import CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, BaseLogger, TensorBoard
train_dir = '/kaggle/input/intel-image-classification/seg_train/seg_train/'

test_dir = '/kaggle/input/intel-image-classification/seg_test/seg_test/'

pred_dir = '/kaggle/input/intel-image-classification/seg_pred/'
print('classes in train_dir: ',os.listdir(train_dir))

print('No of images in prediction folder: ',len(os.listdir(pred_dir + 'seg_pred/')))
diff_classes = os.listdir(train_dir)

diff_classes
train_classes_dir = sum([glob.glob(train_dir + i + '/') for i in diff_classes], [])

train_classes_dir
# os.mkdir('/kaggle/working/augmented_images/')
['No. of images in '+i+': '+str(len(os.listdir(glob.glob(train_dir + i + '/')[0]))) for i in diff_classes]
#To visualize a random image directly call visualize()

#To visualize a particular class image pass that class as a string for obj

def visualize(obj = None):

    if not obj:

        class_dir = random.choice(train_classes_dir)

    else:

        class_dir = train_dir+obj+'/'

    img_path = class_dir + random.choice(os.listdir(class_dir))

    im = cv2.imread(img_path)

    title = plt.title(class_dir.split('/')[-2])

    plt.setp(title, color='r')  

    print(im.shape)

    print(img_path)

    plt.imshow(im[:,:,::-1])
visualize('buildings')
try:

    os.mkdir('/kaggle/working/augmented_images/')

except FileExistsError:

    shutil.rmtree('/kaggle/working/augmented_images/')

    os.mkdir('/kaggle/working/augmented_images/')

    print('Directory Created')

aug_img_dir = '/kaggle/working/augmented_images/'

len(os.listdir(aug_img_dir))
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(directory= train_dir,target_size=(150, 150),batch_size=32,

    color_mode= 'rgb',classes = None,class_mode= "categorical",shuffle=True,seed = 30)

#     save_to_dir = '/kaggle/working/augmented_images/',save_prefix='a',save_format='jpg',interpolation='nearest')



validation_generator = test_datagen.flow_from_directory(directory= test_dir,target_size=(150, 150),batch_size=2,

    color_mode= 'rgb',classes = None,class_mode= "categorical",shuffle=True,seed = 30)



test_generator = test_datagen.flow_from_directory(directory= pred_dir,target_size=(150, 150),color_mode="rgb",

    batch_size=1,class_mode=None,shuffle=False,seed=30)
#Multiclass classification model

ned_stark = Sequential()

ned_stark.add(Conv2D(32, 3, 3, input_shape=(150,150, 3), activation='relu'))

ned_stark.add(MaxPool2D(pool_size=(2, 2)))

ned_stark.add(Conv2D(64, 3, 3, activation='relu'))

ned_stark.add(MaxPool2D(pool_size=(2, 2)))

ned_stark.add(Flatten())

ned_stark.add(Dense(units=128, activation='relu'))

ned_stark.add(Dense(units=6, activation='softmax'))

ned_stark.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(ned_stark.summary())
initial_lr = 0.01

epochs_drop = 2.0 

drop_factor = 0.5



def decay_func(epoch_num):

    new_lr = initial_lr * np.power(drop_factor, epoch_num//epochs_drop)

    return new_lr
# filename =  '/kaggle/working/train.csv'

#Callbacks

csvlogger = CSVLogger(filename='/kaggle/working/training.log',separator = ',', append = False)

earlystopping = EarlyStopping(monitor ='val_loss', min_delta=0.01,patience=2,verbose=1,mode='auto',baseline=None,

                              restore_best_weights =True)

lrscheduler = LearningRateScheduler(schedule= decay_func,verbose= 1)

baselogger = BaseLogger()

checkpointer = ModelCheckpoint(filepath='/kaggle/working/best_model.h5',monitor='val_loss', verbose=1, save_best_only=True,

                               save_weights_only=False, mode='auto', period=1)

tensorboard = TensorBoard(log_dir='/kaggle/working/', batch_size=32)
print(train_generator)

# dir(train_generator)

train_generator.n
model_history = ned_stark.fit_generator(train_generator,steps_per_epoch= train_generator.n//train_generator.batch_size,

epochs=40,validation_data= validation_generator,validation_steps= validation_generator.n//validation_generator.batch_size,

callbacks=[csvlogger, baselogger, checkpointer, tensorboard])



ned_stark.save_weights('/kaggle/working/weights.h5')
len(os.listdir('/kaggle/working/augmented_images/'))
#Evaluate the model

ned_stark.evaluate_generator(generator=validation_generator,steps=validation_generator.n//validation_generator.batch_size)
predictions = ned_stark.predict_generator(test_generator,steps= test_generator.n//test_generator.batch_size,verbose=1)
predictions
predicted_class_indices=np.argmax(predictions,axis=1)

predicted_class_indices
labels = train_generator.class_indices

labels
labels_ = dict((v,k) for k,v in labels.items())

labels_
predictions = [labels_[k] for k in predicted_class_indices]
dir(test_generator)
test_images= test_generator.filenames

test_images
result_df=pd.DataFrame({"Image":test_images,

                      "Predictions":predictions})

result_df.head()
#Visualize a random image from different classes

def visualize_result():

    img_path = pred_dir + random.choice(test_generator.filenames)

    im = cv2.imread(img_path)

    plt.title(result_df.loc[result_df['Image'] == 'seg_pred/' + img_path.split('/')[-1], 'Predictions'].iloc[0])   

    print(im.shape)

    print(img_path)

    plt.imshow(im[:,:,::-1])
visualize_result()
#log file

pd.read_csv('/kaggle/working/training.log')
#csv file

df = pd.read_csv('/kaggle/working/train.csv')

df.head()
dir(my_history)
my_history.history
my_history.params
def visualize_augmented_images():

    img = random.choice(os.listdir(aug_img_dir))

    im = cv2.imread(aug_img_dir + img)

    title_obj = plt.title(img)

    plt.setp(title_obj, color= 'r')

    plt.imshow(im[:,:,::-1])

    plt.show()
visualize_augmented_images()
result_df.to_csv("/kaggle/working/predictions.csv",index=False)
os.listdir('/kaggle/working/')
pd.read_csv('/kaggle/working/predictions.csv').head()
#plot training loss, accuracy
