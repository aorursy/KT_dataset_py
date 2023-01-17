import os

train_dir=os.path.join('/kaggle','working','train')

test_dir = os.path.join('/kaggle','working','test')

os.makedirs(train_dir,exist_ok=True)

os.makedirs(test_dir,exist_ok=True)#Creating a directory for testing examples. 
train_ratio=0.8

test_ratio=0.2

source_dir=os.path.join('/kaggle','input','pins-face-recognition','105_classes_pins_dataset')

dir_list=os.listdir(source_dir)

print(len(dir_list))
#This Code will split up the unzipped dataset into training and testing set with respect to the values of train_ratio and test_ratio.

from shutil import copyfile

import numpy as np

    

dir_list = os.listdir(source_dir)



for folder in dir_list:

    data_dir = os.listdir(os.path.join(source_dir,folder))

    np.random.shuffle(data_dir)                            #Shuffling the examples of the classes inside the dataset.

    os.makedirs(os.path.join(train_dir , folder), exist_ok=True)

    os.makedirs(os.path.join(test_dir , folder), exist_ok=True)

    train_data = data_dir[:int(len(data_dir)*train_ratio+1)] #Splitting the training dataset with respect to the train_ratio.

    test_data = data_dir[-int(len(data_dir)*test_ratio):]

  

    for image in train_data:

        copyfile(os.path.join(source_dir , folder , image),os.path.join( train_dir , folder , image)) #Copying the Training files from dataset to training directory.

    

    for image in test_data:

        copyfile(os.path.join(source_dir , folder , image),os.path.join( test_dir , folder , image))  #Copying the Training files from dataset to training directory.

    


print("The Number of Classes in the Training Set:{}".format(len(os.listdir(train_dir))))

print("The Number Of Classes in the Testing Set:{}".format(len(os.listdir(test_dir))))

print('For Verification of the Split.....')

print("The Number of Examples in the Class='pins_Katherine Langford' in the Dataset Directory:{}".format(len(os.listdir(os.path.join(source_dir,'pins_Katherine Langford')))))

print("The Number of Examples in the Class='pins_Katherine Langford' in the Train Directory:{}".format(len(os.listdir(os.path.join(train_dir,'pins_Katherine Langford')))))

print("The Number of Examples in the Class='pins_Katherine Langford' in the Test Directory:{}".format(len(os.listdir(os.path.join(test_dir,'pins_Katherine Langford')))))

import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

rows=160  #Number of Pixels in the Rows for Input. 

cols=160  #Number of Pixels in Columns for Output
trained_model=MobileNetV2(input_shape=(rows,cols,3),           

                    include_top=False,

                    weights='imagenet')   





trained_model.trainable=True  #Un-Freeze all the pretrained layers of 'MobileNetV2 for Training.

    

trained_model.summary()
last_layer=trained_model.get_layer('out_relu')   

last_layer_output=last_layer.output  #Saves the output of the last layer of the MobileNetV2.
#Callback Function which stops training when accuracy reaches 98%.

class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('accuracy') > 0.98):

      print("\nReached 98% accuracy so cancelling training!")

      self.model.stop_training = True

callbacks=myCallback()
x = tf.keras.layers.GlobalAveragePooling2D()(last_layer_output)

#Add a Dropout layer.

x = tf.keras.layers.Dropout(0.8)(x)                      

# Add a final softmax layer for classification.

x = tf.keras.layers.Dense(105,activation='softmax')(x)



model = tf.keras.Model(trained_model.input,x) 



model.compile(loss='categorical_crossentropy',

              optimizer=tf.keras.optimizers.Adam(1e-4),

              metrics=['accuracy'])

#Summary of the model.

model.summary()
train_datagen = ImageDataGenerator(rescale=1/255,

                                 shear_range=0.2,

                                 zoom_range=0.2,

                                 horizontal_flip=True,

                                 rotation_range=40,

                                 width_shift_range=0.1,

                                 height_shift_range=0.1)

                                 

test_datagen = ImageDataGenerator(rescale=1/255)



train_generator=train_datagen.flow_from_directory(train_dir,

                                                  target_size=(rows,cols),

                                                  class_mode='categorical')



test_generator=test_datagen.flow_from_directory(test_dir,

                                                target_size=(rows,cols),

                                                class_mode='categorical')
result=model.fit(train_generator,

                 validation_data=test_generator,

                 epochs=50,

                 callbacks=[callbacks],

                 verbose=1)
import matplotlib.pyplot as plt



acc = result.history['accuracy']

val_acc = result.history['val_accuracy']



loss = result.history['loss']

val_loss = result.history['val_loss']



epochs_range = range(50)



plt.figure(figsize=(9, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()