# display some example images from the "no" folder (healthy brain MRIs) 



from IPython.display import Image, display

display(Image('../input/brain_tumor_dataset/no/N11.jpg', width=200, height=200))

#display(Image('../input/brain_tumor_dataset/no/No11.jpg', width=200, height=200))

#display(Image('../input/brain_tumor_dataset/no/11 no.jpg', width=200, height=200))

!pip install "tensorflow_hub==0.4.0"
import numpy as np 

import pandas as pd 

import os , shutil

import matplotlib.pyplot as plt

import tensorflow as tf 



from tensorflow import keras



tf.enable_eager_execution()



import tensorflow_hub as hub

from tensorflow.keras import layers

#from tensorflow.keras.applications import mobilenet_v2



print(tf.__version__)

print(keras.__version__)
!ls 
patient_folder = os.listdir('../input/brain_tumor_dataset/yes')

healthy_folder = os.listdir('../input/brain_tumor_dataset/no')


print('some of images in the patient folder', patient_folder[0:10])

print('some of images in the healthy folder', healthy_folder[0:10])
np.random.shuffle(patient_folder)

print('some of images in the patient folder after shuffling', patient_folder[0:10])



np.random.shuffle(healthy_folder)

print('some of images in the patient folder after shuffling', healthy_folder[0:10])
len_training_portion = int(np.floor(0.7*len(patient_folder)))

len_test_portion = int(np.floor(0.15*len(patient_folder)))



train_patient  = patient_folder[0:len_training_portion].copy() 

val_patient = patient_folder[len_training_portion:(len_training_portion + len_test_portion)].copy()

test_patient = patient_folder[(len_training_portion + len_test_portion):].copy()





print('length of the patient folder is {} from which we placed {} for the training, {} for the validation, {} for the test '.

      format(len(patient_folder) , len(train_patient) , len(val_patient), len(test_patient) ) )

len_training_portion = int(np.floor(0.7*len(healthy_folder)))

len_test_portion = int(np.floor(0.15*len(healthy_folder)))



train_healthy  = healthy_folder[0:len_training_portion] 

val_healthy = healthy_folder[len_training_portion:(len_training_portion + len_test_portion)]

test_healthy = healthy_folder[(len_training_portion + len_test_portion):]





print('length of the healthy folder is {} from which we placed {} for the training, {} for the validation, {} for the test '.

      format(len(healthy_folder), len(train_healthy) , len(val_healthy), len(test_healthy) ))

training_data = train_patient + train_healthy

validation_data = val_patient + val_healthy

test_data = test_patient + test_healthy



print('The length of training data after concatenation is ', len(training_data))

print('The length of val data after concatenation is ', len(validation_data))

print('The length of test data after concatenation is ', len(test_data))
# The original directory 

original_dataset_dir_patients = '../input/brain_tumor_dataset/yes'

original_dataset_dir_healthy = '../input/brain_tumor_dataset/no'





# The directory where we will store our dataset

base_dir = 'splitted_data'



# Directories for our training, validation, and test splits

train_dir = os.path.join(base_dir, 'train')

val_dir = os.path.join(base_dir, 'val')

test_dir = os.path.join(base_dir, 'test')



# Directories with our training, validation, testing patient & healthy MRI images





train_patients_dir = os.path.join(train_dir, 'patients')

train_healthy_dir = os.path.join(train_dir, 'healthy')



val_patients_dir = os.path.join(val_dir, 'patients')

val_healthy_dir = os.path.join(val_dir, 'healthy')



test_patients_dir = os.path.join(test_dir, 'patients')

test_healthy_dir = os.path.join(test_dir, 'healthy')

#!rm -r splitted_data
# make the directories



os.mkdir(base_dir)



os.mkdir(train_dir)

os.mkdir(val_dir)

os.mkdir(test_dir)



os.mkdir(train_patients_dir)

os.mkdir(train_healthy_dir)



os.mkdir(val_patients_dir)

os.mkdir(val_healthy_dir)



os.mkdir(test_patients_dir)

os.mkdir(test_healthy_dir)
# check whether the directories are made

print('######################### the root directory')

!ls

print('######################### the created directories')

!ls splitted_data

print('######################### the created directories inside the splitted_data train')

!ls splitted_data/train

print('######################### the created directories inside the splitted_data validation')

!ls splitted_data/val

print('######################### the created directories inside the splitted_data test')

!ls splitted_data/test
def copy_files(fnames , from_dir, to_dir):

    for fname in fnames:

        src = os.path.join(from_dir, fname)

        dst = os.path.join(to_dir, fname)

        shutil.copyfile(src, dst)



        

copy_files(train_patient, original_dataset_dir_patients , train_patients_dir)

copy_files(train_healthy, original_dataset_dir_healthy , train_healthy_dir)



copy_files(val_patient, original_dataset_dir_patients , val_patients_dir)

copy_files(val_healthy, original_dataset_dir_healthy , val_healthy_dir)



copy_files(test_patient, original_dataset_dir_patients , test_patients_dir)

copy_files(test_healthy, original_dataset_dir_healthy , test_healthy_dir)



# do a sanity check on the length of the data stored in each directory



print('Length of the data stored in the training directory (train_dir) is: ', len(os.listdir(train_dir) ) )



print('Length of the data stored in the training directory of patients (train_patients_dir) is: ', len(os.listdir(train_patients_dir) ) )



print('Length of the data stored in the training directory of healthy subjects (train_healthy_dir) is: ', len(os.listdir(train_healthy_dir) ) )







print('Length of the data stored in the validation directory (val_dir) is: ', len(os.listdir(val_dir) ) )



print('Length of the data stored in the validation directory of patients (val_patients_dir) is: ', len(os.listdir(val_patients_dir) ) )



print('Length of the data stored in the validation directory of healthy subjects (val_healthy_dir) is: ', len(os.listdir(val_healthy_dir) ) )







print('Length of the data stored in the test directory (test_dir) is: ', len(os.listdir(test_dir) ) )



print('Length of the data stored in the test directory of patients (test_patients_dir) is: ', len(os.listdir(test_patients_dir) ) )



print('Length of the data stored in the test directory of healthy subjects (test_healthy_dir) is: ', len(os.listdir(test_healthy_dir) ) )
img_res = 224 # if we are using MobileNet_V2





### initialize the image data generator for training data



image_gen_train = keras.preprocessing.image.ImageDataGenerator(

      rescale=1./255,

      rotation_range=40,

      horizontal_flip=True,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      fill_mode='nearest')





### flow the training data from directory while they are resized 



train_generator = image_gen_train.flow_from_directory(

        train_dir, target_size=(img_res, img_res), class_mode='binary')







### initialize the image data generator for validation data



image_gen_val = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,

      rotation_range=40,

      horizontal_flip=True,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      fill_mode='nearest')





### flow the validation data from directory while they are resized 



val_generator = image_gen_val.flow_from_directory(

        val_dir, target_size=(img_res, img_res), class_mode='binary')



##### MobileNet Version 2 from keras  



#feature_extractor = mobilenet_v2.MobileNetV2(input_shape=(img_res, img_res, 3), include_top=False, weights='imagenet', classes=2)



##### we can also use tensorflow_hub for a pretrained MobileNet V2 



URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL, input_shape=(img_res, img_res,3))
feature_extractor.trainable = False



def create_model():

  model = keras.Sequential()

  model.add(feature_extractor)

  

  model.add(layers.BatchNormalization())

  

  model.add(layers.Dense(128, activation = 'relu'))

  model.add(layers.Dropout(0.5))

  

  model.add(layers.Dense(1, activation='sigmoid'))



  return model



model = create_model()
model.summary()
           

model.compile(

  optimizer=keras.optimizers.Adam(),

  loss='binary_crossentropy',

  metrics=['accuracy' ])
filepath="the_best_model.ckpt"

checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

 

history = model.fit_generator(

      train_generator,

      steps_per_epoch=30,

      epochs=50,

      validation_data=val_generator,

      validation_steps=30,

      callbacks = [checkpoint])
def plot_history(history):

  

  acc = history.history['acc']

  val_acc = history.history['val_acc']



  loss = history.history['loss']

  val_loss = history.history['val_loss']



  epochs_range = np.arange(len(history.history['loss']))



  plt.figure(figsize=(10, 5))

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

  #plt.savefig('./foo.png')

  plt.show()

  

########################

  

plot_history(history)
###### loading the best model

#best_model = tf.keras.experimental.load_from_saved_model("the_best_model.ckpt", custom_objects={'KerasLayer':hub.KerasLayer})

best_model = keras.models.load_model("the_best_model.ckpt" , custom_objects={'KerasLayer':hub.KerasLayer})



###### initializing a test data generator



test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory(

        test_dir, target_size=(img_res, img_res), batch_size=40, class_mode='binary')



test_loss, test_acc = best_model.evaluate_generator(test_generator, steps=1)

print('test acc:', test_acc)