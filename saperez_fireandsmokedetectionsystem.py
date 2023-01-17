#%tensorflow_version 2.x
import tensorflow as tf
import timeit
#tf.config.experimental.list_physical_devices('GPU')
import os
import math
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
import timeit
import time
import zipfile
import urllib.request

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

!pip3 install kaggle

#Download requiered datasets from different data sources
# ! mkdir ~/.kaggle
# ! cp kaggle.json ~/.kaggle/
# ! chmod 600 ~/.kaggle/kaggle.json

# ! kaggle datasets list
# ! kaggle datasets download -d phylake1337/fire-dataset

#Download and import test data
def create_folder(path, folder_name):
  if ('images' not in os.listdir(os.getcwd())):
    os.mkdir(path)
  if (folder_name not in os.listdir(path)):
    os.mkdir(path + '/' + folder_name)

    

create_folder(os.getcwd()+'/images', 'train')
create_folder(os.getcwd()+'/images', 'test')
create_folder(os.getcwd()+'/images', 'val')

create_folder(os.getcwd()+'/images/train', 'fire')
create_folder(os.getcwd()+'/images/train', 'smoke')
create_folder(os.getcwd()+'/images/train', 'neutral')

create_folder(os.getcwd()+'/images/test', 'fire')
create_folder(os.getcwd()+'/images/test', 'smoke')
create_folder(os.getcwd()+'/images/test', 'neutral')

create_folder(os.getcwd()+'/images/val', 'fire')
create_folder(os.getcwd()+'/images/val', 'smoke')
create_folder(os.getcwd()+'/images/val', 'neutral')

train_images_path = os.getcwd() + '/images/train'
test_images_path = os.getcwd() + '/images/val'
create_folder(os.getcwd(),'resources')



postive_cases = 'fire'
negative_cases = 'non_fire'


!wget --quiet -O test_data_keras.zip https://github.com/DeepQuestAI/Fire-Smoke-Dataset/releases/download/v1/FIRE-SMOKE-DATASET.zip
# !wget --quiet -O fire_dataset_kaggle.zip https://www.kaggle.com/phylake1337/fire-dataset/download
# urllib.request.urlretrieve('https://www.kaggle.com/phylake1337/fire-dataset/download')
with zipfile.ZipFile('test_data_keras.zip', 'r') as zip_ref:
    zip_ref.extractall('resources')
# with zipfile.ZipFile('fire_dataset_kaggle.zip', 'r') as zip_ref:
#     zip_ref.extractall('resources')    
    
#Move files to the final folder
fire_dataset_fire= '/kaggle/input/fire-dataset/fire_dataset/fire_images'
fire_dataset_neutral = '/kaggle/input/fire-dataset/fire_dataset/non_fire_images'
fire_dataset2_path = os.getcwd()+'/resources/FIRE-SMOKE-DATASET/Train/Fire'

train_fire_data = 'images/train/fire'
train_neutral_data = 'images/train/neutral'
train_smoke_data = 'images/train/smoke'

test_fire_data = 'images/test/fire'
test_neutral_data = 'images/test/neutral'
test_smoke_data = 'images/test/smoke'

positive_samples = os.listdir(fire_dataset_fire)
neutral_samples = os.listdir(fire_dataset_neutral)

positive_samples_ds2 = os.listdir(fire_dataset2_path)



ratio = 0.8
if (math.ceil(len(positive_samples)*ratio) > math.ceil(len(neutral_samples)*ratio)):
  img_to_move_from_kaggle = math.ceil(len(neutral_samples)*ratio)
else:
  img_to_move_from_kaggle = math.ceil(len(positive_samples)*ratio)

img_to_move_from_ds2 = len(positive_samples)

dataset_2_path = os.getcwd()+'/resources/FIRE-SMOKE-DATASET'
dataset_1_path ='/kaggle/input/fire-dataset/fire_dataset'

for folder in os.listdir(dataset_2_path):
  if folder=='Train':
    fire_images = os.listdir(os.path.join(dataset_2_path,folder,'Fire'))
    neutral_images = os.listdir(os.path.join(dataset_2_path,folder,'Neutral'))
    smoke_images = os.listdir(os.path.join(dataset_2_path,folder,'Smoke'))
    for fire, smoke, neutral in zip(fire_images, neutral_images, smoke_images):
      shutil.move(os.path.join(dataset_2_path, folder,'Fire',fire), os.path.join(train_fire_data))
      shutil.move(os.path.join(dataset_2_path, folder,'Smoke',smoke), os.path.join(train_smoke_data))
      shutil.move(os.path.join(dataset_2_path, folder,'Neutral',neutral), os.path.join(train_neutral_data))
  if folder=='Test':
    fire_images = os.listdir(os.path.join(dataset_2_path,folder,'Fire'))
    neutral_images = os.listdir(os.path.join(dataset_2_path,folder,'Neutral'))
    smoke_images = os.listdir(os.path.join(dataset_2_path,folder,'Smoke'))
    for fire, smoke, neutral in zip(fire_images, neutral_images, smoke_images):
      shutil.move(os.path.join(dataset_2_path, folder,'Fire',fire), os.path.join(test_fire_data))
      shutil.move(os.path.join(dataset_2_path, folder,'Smoke',smoke), os.path.join(test_neutral_data))
      shutil.move(os.path.join(dataset_2_path, folder,'Neutral',neutral), os.path.join(test_smoke_data))


count = 0
fire_images = os.listdir(os.path.join(dataset_1_path,'fire_images'))
neutral_images = os.listdir(os.path.join(dataset_1_path,'non_fire_images'))
for fire, neutral in zip(fire_images, neutral_images):
    count += 1
    if (count<img_to_move_from_kaggle):
        pass
        shutil.copyfile(os.path.join(dataset_1_path, 'fire_images',fire), os.path.join(train_fire_data,fire))
        shutil.copyfile(os.path.join(dataset_1_path,'non_fire_images',neutral), os.path.join(train_neutral_data,neutral))

image = load_img(os.getcwd() + '/images/train/neutral/image_110.jpg')
plt.imshow(image)
plt.axis('off')

validation_data_dir = os.getcwd() + '/images/test'
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        rotation_range=30,
        fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

val_generator_resnet = test_datagen.flow_from_directory(
    validation_data_dir,
    batch_size=32,
    target_size=(224,224),
    classes=['fire', 'neutral','smoke'],
    class_mode='categorical'
)
val_generator_xception = test_datagen.flow_from_directory(
    validation_data_dir,
    batch_size=32,
    target_size=(299, 299),
    classes=['fire', 'neutral','smoke'],
    class_mode='categorical'
)
train_generator_restnet = train_datagen.flow_from_directory(
        # np.expand_dims(img_to_array(load_img('./images/fire_images/fire.662.png')), axis=0),
        train_images_path,
        target_size=(224, 224),
        batch_size=32, 
        # save_to_dir='./images/augmented_images', 
        # save_prefix='keras_',
        # save_format='png',
        classes=['fire', 'neutral','smoke'],
        class_mode='categorical'
)

train_generator_xception = train_datagen.flow_from_directory(
        # np.expand_dims(img_to_array(load_img('./images/fire_images/fire.662.png')), axis=0),
        train_images_path,
        target_size=(299, 299),
        batch_size=32, 
        # save_to_dir='./images/augmented_images', 
        # save_prefix='keras_',
        # save_format='png',
        classes=['fire', 'neutral','smoke'],
        class_mode='categorical'
)

count = 0

image_batch = train_generator_restnet.next()
f, ax = plt.subplots(2,3, figsize=(20,11))
for i, img in enumerate(image_batch[0]):
    count += 1
    if (count > 6):
        break
    ax[i//3, i%3].imshow(img)
    ax[i//3, i%3].axis('off')
    label = np.argmax(image_batch[1][i], axis=0)
    ax[i//3, i%3].title.set_text(str(label))
plt.show()

from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from keras import regularizers
from keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K

#Instantiate pre-trained models for feature extraction and classification
#Models list.
    #Resnet50: Size=98MB, TOP-1 Accuracy=0.749, # Parameters=25,636,712, inpuit_dim=(224,224)
    #Xception: Size=88MB, TOP-1 Accuracy=0.790, # Parameters=22,910,480, input_dim=(299,299)
rn_base = ResNet50(weights='imagenet')
xc_base = Xception(weights='imagenet')

for layer in rn_base.layers:
    layer.trainable=False
for layer in xc_base.layers:
    layer.trainable=False    
NUM_CLASSES = 3
#Select last hidden layer's output
sec_last_base_rs = rn_base.layers[-2].output
sec_last_base_xc = xc_base.layers[-2].output

x_rn = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(sec_last_base_rs)
x_rn = Activation('relu')(x_rn)
x_rn = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x_rn)
x_rn = Activation('relu')(x_rn)
rn_connected_model = Dense(NUM_CLASSES, activation='softmax')(x_rn)

x_xc = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(sec_last_base_xc)
x_xc = Activation('relu')(x_xc)
x_xc = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x_xc)
x_xc = Activation('relu')(x_xc)
xc_connected_model = Dense(NUM_CLASSES, activation='softmax')(x_xc)

#Get the model's input tensor
base_input_rs = rn_base.input
base_input_xc = xc_base.input
#Assemble the final model
rn_model = Model(inputs=base_input_rs, outputs=rn_connected_model)
xc_model = Model(inputs=base_input_xc, outputs=xc_connected_model)
#Compile models
rn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
xc_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#%%
#Construct the callback functions
early_stop = EarlyStopping(monitor='val_loss', patience=9, verbose=1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, min_delta=1e-4)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='models/model.{epoch:02d}-{val_loss:.2f}.h5'),
model_checkpoint_xc = tf.keras.callbacks.ModelCheckpoint(filepath='XceptionModel/model.{epoch:02d}-{val_loss:.2f}.h5'),
callbacks_list = [early_stop, reduce_lr]
callbacks_list_xc = [early_stop, reduce_lr]
#Train resnet model
N_EPOCHS = 2
STEPS_PER_EPOCH = train_generator_restnet.n // train_generator_restnet.batch_size

start_rs = time.time()
print('[INFO] Training Model Resnet.. \n')
rn_history = rn_model.fit(train_generator_restnet,
                                    validation_data=val_generator_xception,  
                                    validation_steps=val_generator_xception.n//32,
                                    epochs=N_EPOCHS,
                                    steps_per_epoch=STEPS_PER_EPOCH,
                                    callbacks=callbacks_list)
print('[INFOR] Resnet model training finished. \n')
end_rs = time.time()
print('Time elapesed: ', end_rs-start_rs)
start = timeit.timeit()
print('[INFO] Training Model Xception.. \n')
xc_history = xc_model.fit(train_generator_xception,
                          validation_data=val_generator_xception,  
                          validation_steps=val_generator_xception.n//32,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          epochs=N_EPOCHS,
                          callbacks=callbacks_list_xc)
print('[INFOR] Xception model training finished. \n')
end = timeit.timeit()
print('Time elapesed: ', end-start)

def plot_train_accuracy(histories, model_names):
  plt.figure(figsize=(5,5))
  fig,ax = plt.subplots()
  for history, m_name in zip(histories, model_names):
    hor_axis = np.arange(0, len(history.history['accuracy']))
    plt.plot(hor_axis, history.history['accuracy'], label='{} Training Accuracy'.format(m_name))
    plt.xlabel('EPOCHS')
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, len(history.history['accuracy']), 1))
    plt.legend()
  plt.show()

# print('Resnet Training Accuracy: ', rn_history.history['accuracy'][-1])
plot_train_accuracy([rn_history, xc_history], ['Resnet', 'Xception'])

