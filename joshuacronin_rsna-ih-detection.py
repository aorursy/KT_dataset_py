# Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from os import listdir

from os.path import isfile, join

import matplotlib.pylab as plt

import os

import seaborn as sns

from tqdm import tqdm



from keras.applications import DenseNet121

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.initializers import Constant

from keras.models import Sequential

from keras.optimizers import Adam

from keras import layers



import os





# Any results you write to the current directory are saved as output.
INPUT_PATH = "../input/rsna-intracranial-hemorrhage-detection/"

TRAIN_DIRECTORY = 'stage_1_train_images/'

TEST_DIRECTORY = 'stage_1_test_images/'
train_dataframe = pd.read_csv(INPUT_PATH + "stage_1_train.csv")

train_dataframe.head()
label = train_dataframe.Label.values
train_dataframe['filename'] = train_dataframe['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")

train_dataframe['type'] = train_dataframe['ID'].apply(lambda st: st.split('_')[2])
train_dataframe.head()
pivot_df = train_dataframe[['Label', 'filename', 'type']].drop_duplicates().pivot(

    index='filename', columns='type', values='Label').reset_index()

print(pivot_df.shape)

pivot_df.head()
train_dir = INPUT_PATH + "stage_1_train_images/"

train_files = os.listdir(train_dir)

train_size = len(train_files)

train_size
#Get the training image directory

train_images_directory = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'

#get the training images

train_images = [file for file in listdir(train_images_directory) if isfile(join(train_images_directory,file))]

#repeat for test images

test_images_directory = '../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'

test_images = [file for file in listdir(test_images_directory) if isfile(join(test_images_directory, file))]

#check some image filenames

print('5 Training Image Files', train_images[:5] )

np.random.seed(42)

sample_files = np.random.choice(train_files, 100000)

sample_df = pivot_df[pivot_df.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_files)]
def window_image(img, window_center,window_width, intercept, slope, rescale=True):



    img = (img*slope +intercept)

    img_min = window_center - window_width//2

    img_max = window_center + window_width//2

    img[img<img_min] = img_min

    img[img>img_max] = img_max

    

    if rescale:

        # Extra rescaling to 0-1, not in the original notebook

        img = (img - img_min) / (img_max - img_min)

    

    return img

    

def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
from tqdm import tqdm

import pydicom

import cv2

def save_and_resize(filenames, load_dir):    

    save_dir = '/kaggle/tmp/'

    if not os.path.exists(save_dir):

        os.makedirs(save_dir)



    for filename in tqdm(filenames):

        path = load_dir + filename

        new_path = save_dir + filename.replace('.dcm', '.png')

        

        dcm = pydicom.dcmread(path)

        window_center , window_width, intercept, slope = get_windowing(dcm)

        img = dcm.pixel_array

        img = window_image(img, window_center, window_width, intercept, slope)

        

        resized = cv2.resize(img, (224, 224))

        res = cv2.imwrite(new_path, resized)
#to save time with the commit I have commented this step out - it takes some time

#TODO : Speed this up or look at working directly with the dicom images? 



save_and_resize(filenames=sample_files, load_dir=INPUT_PATH + TRAIN_DIRECTORY)

#save_and_resize(filenames=os.listdir(INPUT_PATH + TEST_DIRECTORY), load_dir=INPUT_PATH + TEST_DIRECTORY)
#Lets try using DenseNet 121



densenet= DenseNet121(

    weights = None,

    include_top= False,

    input_shape=(224,224,3)

)
def build_dense_model():

    model = Sequential()

    model.add(densenet)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(6, activation='sigmoid', 

                           bias_initializer=Constant(value=-5.5)))

    

    model.compile(

        loss='categorical_crossentropy',

        optimizer=Adam(lr=0.001),

        metrics=['accuracy']

    )

    

    return model
model = build_dense_model()

model.summary()
#This datagen is used for feeding in the train data. 



BATCH_SIZE = 64



def create_datagen():

    return ImageDataGenerator(validation_split=0.15)





def create_flow(datagen, subset):

    return datagen.flow_from_dataframe(

        pivot_df, 

        directory='/kaggle/tmp/',

        x_col='filename', 

        y_col=['any', 'epidural', 'intraparenchymal', 

               'intraventricular', 'subarachnoid', 'subdural'],

        class_mode='multi_output',

        target_size=(224, 224),

        batch_size=BATCH_SIZE,

        subset=subset

    )



# Using original generator

data_generator = create_datagen()

train_gen = create_flow(data_generator, 'training')

val_gen = create_flow(data_generator, 'validation')

checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_loss', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



total_steps = sample_files.shape[0] / BATCH_SIZE



history = model.fit_generator(

    train_gen,

    steps_per_epoch=2000,

    validation_data=val_gen,

    validation_steps=total_steps * 0.15,

    callbacks=[checkpoint],

    epochs=5

)