# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time, gc

import tensorflow as tf

from PIL import Image

print(tf.__version__)



from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import matplotlib

matplotlib.use('Agg')



# import the necessary keras and sklearn packages



from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split



import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!ls /kaggle/input/
from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path('bengaliai-cv19')
!gsutil ls $GCS_DS_PATH
GCS_DS_PATH1 = KaggleDatasets().get_gcs_path('bengaliaicv19feather')
!gsutil ls $GCS_DS_PATH1
!pip install gcsfs
train_df_ = pd.read_csv('gs://kds-06929b980722279141ec67c5f76a6468a973cd72023b750636bda6c7/train.csv')

class_map_df = pd.read_csv('gs://kds-06929b980722279141ec67c5f76a6468a973cd72023b750636bda6c7/class_map.csv')
print(train_df_.head())
len(train_df_)
print(class_map_df.head())
print(class_map_df.component_type.value_counts())
class_map_df_root = class_map_df[class_map_df.component_type=='grapheme_root']

class_map_df_vowel = class_map_df[class_map_df.component_type=='vowel_diacritic']

class_map_df_cons = class_map_df[class_map_df.component_type=='consonant_diacritic']
graphemeLB = LabelBinarizer()

vowelLB = LabelBinarizer()

consonantLB = LabelBinarizer()



graphemeLB.fit(class_map_df_root.label)

vowelLB.fit(class_map_df_vowel.label)

consonantLB.fit(class_map_df_cons.label)
print(len(vowelLB.classes_))

print(len(consonantLB.classes_))

print(len(graphemeLB.classes_))
def read_data(nf):

    nf=int(nf)

    train_df = pd.read_parquet(f'gs://kds-06929b980722279141ec67c5f76a6468a973cd72023b750636bda6c7/train_image_data_{nf}.parquet')

     #   f'gs://kds-87e1f7817c6764d20c2f2841fd9048ac1f7b9c89a1508dbd796f13b4/train_image_data_{nf}.feather')

     #   f'/kaggle/input/bengaliaicv19feather/train_image_data_{nf}.feather')

     #   f'gs://kds-06929b980722279141ec67c5f76a6468a973cd72023b750636bda6c7/train_image_data_{nf}.parquet'

    return train_df
def res_net_block_1(input_data, filters):

  

    x1 = tf.keras.layers.Conv2D(filters, 3, activation=tf.nn.relu, padding='same')(input_data)

    x1 = tf.nn.leaky_relu(x1, alpha=0.01, name='Leaky_ReLU') 

    x2 = tf.keras.layers.BatchNormalization()(x1)

    x2 = tf.keras.layers.Dropout(0.3)(x2)

    

    x3 = tf.keras.layers.Conv2D(filters, 5, activation=None, padding='same')(x2)

    x3 = tf.nn.leaky_relu(x3, alpha=0.01, name='Leaky_ReLU') 

    x4 = tf.keras.layers.BatchNormalization()(x3)

    x4 = tf.keras.layers.Dropout(0.3)(x4)

  

    x5 = tf.keras.layers.Conv2D(filters, 1, activation=None, padding='same')(input_data)

    x5 = tf.nn.leaky_relu(x5, alpha=0.01, name='Leaky_ReLU') 



    x = tf.keras.layers.Add()([x4 , x5 ])

    x = tf.keras.layers.Activation(tf.nn.relu)(x)

    return x
def res_net_block_2(input_data, filters):

  

    x1 = tf.keras.layers.Conv2D(filters, 3, activation=tf.nn.relu, padding='same')(input_data)

    x1 = tf.nn.leaky_relu(x1, alpha=0.01, name='Leaky_ReLU') 

    x2 = tf.keras.layers.BatchNormalization()(x1)

    x2 = tf.keras.layers.Dropout(0.3)(x2)

    

    x3 = tf.keras.layers.Conv2D(filters, 5, activation=None, padding='same')(input_data)

    x3 = tf.nn.leaky_relu(x3, alpha=0.01, name='Leaky_ReLU') 

    x4 = tf.keras.layers.BatchNormalization()(x3)

    x4 = tf.keras.layers.Dropout(0.3)(x4)

  

    x5 = tf.keras.layers.Conv2D(filters, 1, activation=None, padding='same')(input_data)

    x5 = tf.nn.leaky_relu(x5, alpha=0.01, name='Leaky_ReLU') 



    x = tf.keras.layers.Add()([x2 , x4 , x5 ])

    x = tf.keras.layers.Activation(tf.nn.relu)(x)

    return x
def resnet(inputsize,outputsize,depth,model_type):

    inputs = tf.keras.layers.Input(shape=(inputsize,inputsize,1))

    x = tf.keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu)(inputs)

    x = tf.nn.leaky_relu(x, alpha=0.01, name='Leaky_ReLU') 

    x = tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu)(x)

    x = tf.nn.leaky_relu(x, alpha=0.01, name='Leaky_ReLU') 

    x = tf.keras.layers.MaxPooling2D(3)(x)

    x = tf.keras.layers.Dropout(0.1)(x)

    num_res_net_blocks = depth

    for i in range(num_res_net_blocks):

        x = res_net_block_2(x, 64)

    x = tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu)(x)

    x = tf.nn.leaky_relu(x, alpha=0.01, name='Leaky_ReLU') 

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)

    if (model_type != "root"):

        x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)

        x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)

        x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(x)

    x = tf.keras.layers.Dropout(0.7)(x)

    output = tf.keras.layers.Dense(outputsize, activation=tf.nn.softmax)(x)

    model = tf.keras.models.Model(inputs, output)

    return model



AUTO = tf.data.experimental.AUTOTUNE

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
ResNet = True 

CNN = False

with strategy.scope():

    model_root = resnet(96, 168,10,"root")  # Input imagesize, outputtensor size, depth

    #model_vowel = resnet(96, 11,10,"vowel")

    #model_consonant = resnet(96, 7,10,"consonant")
tf.keras.utils.plot_model(model_root, to_file='model1.png')

#tf.keras.utils.plot_model(model_vowel, to_file='model2.png')

#tf.keras.utils.plot_model(model_consonant, to_file='model3.png')
EPOCHS = 2

INIT_LR = 1e-3

BS = 128

# initialize the optimizer and compile the model

print("[INFO] compiling models...")

opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model_root.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

#model_vowel.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

#model_consonant.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
histories = []



es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)



for i in range(2):

    print("iteration:"+str(i))



    graphemeLabels = []

    vowelLabels = []

    consonantLabels = []   

    print("[INFO] reading train images and labels...")

    train_df = pd.merge(read_data(i), train_df_, on='image_id').drop(['image_id','grapheme'], axis=1)[1:1000]

    graphemeLabels = train_df.grapheme_root

    vowelLabels = train_df.vowel_diacritic

    consonantLabels = train_df.consonant_diacritic



    print("[INFO] binarizing labels...")

    graphemeLabels = graphemeLB.transform(np.array(graphemeLabels))

    vowelLabels = vowelLB.transform(np.array(vowelLabels))

    consonantLabels = consonantLB.transform(np.array(consonantLabels))



    print(graphemeLabels.shape)

    print(vowelLabels.shape)

    print(consonantLabels.shape)



    train_df=train_df.drop(["consonant_diacritic","grapheme_root","vowel_diacritic"],axis=1)

    

    print("[INFO] doing train test split...")

    (trainX, testX, trainGraphemeY, testGraphemeY,trainVowelY, testVowelY,trainConsonantY,testConsonantY) = train_test_split(train_df, graphemeLabels, vowelLabels,consonantLabels,test_size=0.01, random_state=42)

   

    del train_df

    del graphemeLabels

    del vowelLabels

    del consonantLabels

    gc.collect()



    print("[INFO] creating train dataset...")

    trainX=np.array(trainX).reshape(-1,137,236,1)

    print(trainX.shape)

    resized_image=[]

    for j in range(trainX.shape[0]):

        resized_img = tf.image.resize(trainX[j],[96,96])

        resized_img=np.array(resized_img)/255.

        resized_image.append(resized_img)

    resized_image = np.asarray(resized_image)

    resized_image = tf.cast(resized_image, tf.int32)

    trainGraphemeY = tf.cast(trainGraphemeY, tf.int32)

    print(resized_image.shape)

    del trainX

    gc.collect()

    

    print("[INFO] Creating Augmented Images...")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(

            featurewise_center=False,  # set input mean to 0 over the dataset

            samplewise_center=False,  # set each sample mean to 0

            featurewise_std_normalization=False,  # divide inputs by std of the dataset

            samplewise_std_normalization=False,  # divide each input by its std

            zca_whitening=False,  # apply ZCA whitening

            rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)

            zoom_range = 0.15, # Randomly zoom image 

            width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

            height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

            horizontal_flip=False,  # randomly flip images

            vertical_flip=False)  # randomly flip images



    datagen.fit(resized_image)

    datagen = datagen.flow(resized_image, trainGraphemeY, batch_size=BS)

    

    print("[INFO] Creating TF Dataset...")

    ds = tf.data.Dataset.from_generator(

        lambda:datagen,

    output_types=(tf.int32, tf.int32),

    output_shapes=(resized_image.shape, trainGraphemeY.shape)

    )  

    

    #filename = 'test.tfrecord'

    #writer = tf.io.TFRecordWriter(filename)

    #writer.write(ds)

    

    #dataset = tf.data.TFRecordDataset(ds)

    

    #print("[INFO] creating validation dataset...")

    #testX=np.array(testX).reshape(-1,137,236,1)

    #print(testX.shape)

    #resized_image_test=[]

    #for i in range(len(testX)):

    #    resized_img = tf.image.resize(testX[i],[96,96])

    #    resized_img=np.array(resized_img)/255.

    #    resized_image_test.append(resized_img)

    #resized_image_test = np.asarray(resized_image_test)



    #del testX

    #gc.collect()



    print("[INFO] Root Model.fit starting...")



    history = model_root.fit_generator(ds,

                                      epochs = EPOCHS, 

                                      steps_per_epoch=resized_image.shape[0] // BS, 

                                      callbacks=[es],verbose=2)

    #validation_data = (resized_image_test,testGraphemeY)

    histories.append(history)

    

    #print("[INFO] Vowel Model.fit starting...")

    #history = model_vowel.fit_generator(datagen.flow(resized_image, trainVowelY, batch_size=BS),

    #                                  epochs = EPOCHS, validation_data = (resized_image_test,testVowelY),

    #                                  steps_per_epoch=resized_image.shape[0] // BS, 

    #                                  callbacks=[es],verbose=2)



    #histories.append(history)

    

    #print("[INFO] Cons Model.fit starting...")

    #history = model_consonant.fit_generator(datagen.flow(resized_image, trainConsonantY, batch_size=BS),

    #                                  epochs = EPOCHS, validation_data = (resized_image_test,testConsonantY),

    #                                  steps_per_epoch=resized_image.shape[0] // BS, 

    #                                  callbacks=[es],verbose=2)



    #histories.append(history)

    

    #del resized_image

    #del resized_image_test

    #gc.collect()