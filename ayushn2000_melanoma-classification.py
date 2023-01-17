# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# !pip install tf-nightly

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import cv2

from sklearn.model_selection import train_test_split

from kaggle_datasets import KaggleDatasets



AUTOTUNE = tf.data.experimental.AUTOTUNE

GCS_path = KaggleDatasets().get_gcs_path('melanoma-256x256')

GCS_path
dataframe = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')



dataframe['image_path'] ='../input/siim-isic-melanoma-classification/jpeg/train/' + dataframe['image_name']+'.jpg'

dataframe.pop('image_name')



dataframe['anatom_site'] = pd.Categorical(dataframe['anatom_site_general_challenge'])

anatom_site_categories = dataframe.anatom_site.cat.categories

dataframe['anatom_site'] = dataframe.anatom_site.cat.codes

dataframe['anatom_site'].fillna((dataframe['anatom_site'].mean()), inplace=True)



dataframe['sex'] = pd.Categorical(dataframe['sex'])

sex_categories = dataframe.sex.cat.categories

dataframe['sex'] = dataframe.sex.cat.codes

dataframe['sex'].fillna((dataframe['sex'].mean()), inplace=True)



dataframe['age_approx'].fillna((dataframe['age_approx'].mean()), inplace=True)

dataframe = dataframe.astype({"sex":np.int32,"age_approx":np.int32, 'target':np.int8})



dataframe = dataframe.drop(columns=['diagnosis', 'patient_id', 'anatom_site_general_challenge'])

dataframe.to_csv('dataframe.csv')
print("Anatom_sites:")

for idx, anatom_site in enumerate(anatom_site_categories):

    print(str(idx)+": "+anatom_site, end='\n')



print("\nSex:")

for idx, sex in enumerate(sex_categories):

    print(str(idx)+": "+sex, end='\n')
dataframe.sample(5)
dataframe.benign_malignant.value_counts()
pos_values = dataframe['target'].values != 0

pos_dataframe = dataframe.iloc[pos_values, :]



plt.figure(figsize=(20,8))





for i in range(24):

    img = cv2.imread(pos_dataframe.iloc[i,4])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax = plt.subplot(4,6,i+1)

    plt.imshow(img)

    plt.axis('off')

neg_dataframe =  dataframe.iloc[~pos_values, :]



plt.figure(figsize=(20,8))





for i in range(24):

    img = cv2.imread(neg_dataframe.iloc[i,4])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax = plt.subplot(4,6,i+1)

    plt.imshow(img)

    plt.axis('off')
plt.suptitle("Distribution of Melanoma patient skin images with respect to age ")

plt.hist(pos_dataframe['age_approx'].values,bins=15)

plt.show()
plt.suptitle("Distribution of Non-Melanoma patient skin images with respect to age")

plt.hist(neg_dataframe['age_approx'].values,bins=15)

plt.show()
plt.bar(dataframe['anatom_site'].value_counts().keys(),dataframe['anatom_site'].value_counts().values)

plt.suptitle('Occurence of images of various body parts, of both targets')
BATCH_SIZE = 10

EPOCHS = 10

validation_split = 0.15



files_train = tf.io.gfile.glob(GCS_path+'/train*.tfrec')

files_test = tf.io.gfile.glob(GCS_path+'/test*.tfrec')



split = int(len(files_train) * validation_split)

validation_filenames = files_train[:split]

training_filenames = files_train[split:]

print("Pattern matches {} data files. Splitting dataset into {} training files and {} validation files".format(len(files_train), len(training_filenames), len(validation_filenames)))

@tf.function

def read_tfrecord(example):

    features = {

        'image'                        : tf.io.FixedLenFeature([], tf.string),

        'image_name'                   : tf.io.FixedLenFeature([], tf.string),

        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),

        'sex'                          : tf.io.FixedLenFeature([], tf.int64),

        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),

        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),

        'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),

        'target'                       : tf.io.FixedLenFeature([], tf.int64)

    }

    example = tf.io.parse_single_example(example, features)

    

    image = tf.image.decode_jpeg(example["image"], channels=3)

    image = tf.cast(image, dtype= 'float32') / 255.0

    image = tf.image.resize(image, [256,256])

    

    return (image, 

            example['sex'],

            example['age_approx'],

            example['anatom_site_general_challenge']

           ), example['target']



@tf.function

def augment(data, target):

    img = data[0]

    img = tf.image.random_brightness(img, 0.25)

    img = tf.image.random_contrast(img, 0.5, 0.6)

    img = tf.image.random_flip_left_right(img)

    img = tf.image.random_flip_up_down(img)

    img = tf.image.random_hue(img, 0.02)

    # img = tf.image.random_jpeg_quality(img, 85, 100)

    img = tf.cast(img, tf.float32)

    

    return (img, data[1], data[2], data[3]), target





def load_dataset(filenames):

    # read from TFRecords

    option_no_order = tf.data.Options()

    option_no_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)

    dataset = dataset.with_options(option_no_order)

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)

    return dataset
def load_batched_dataset(filenames, training=True):

    dataset = load_dataset(filenames)

    if training:

        dataset = dataset.repeat()

    dataset = dataset.shuffle(buffer_size= 100, reshuffle_each_iteration=True)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset



training_dataset = load_batched_dataset(training_filenames)

validation_dataset = load_batched_dataset(validation_filenames, training=False)
neg, pos = np.bincount(dataframe['target'])

total = len(dataframe)

print("negative (benign): "+str(neg))

print("potitive (malignant): "+str(pos))

print("total examples: "+str(total))
input_1 = tf.keras.Input(shape=(256,256,3), name='image')

input_2 = tf.keras.Input(shape=(1), name='sex')

input_3 = tf.keras.Input(shape=(1), name='age_approx')

input_4 = tf.keras.Input(shape=(1), name='anatom_site')



cnn1 = tf.keras.applications.densenet.DenseNet169(input_shape=(256,256,3), include_top=False, weights=None)(input_1)

cnn1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu', use_bias=True)(cnn1)

cnn1 = tf.keras.layers.Flatten()(cnn1)

cnn1 = tf.keras.layers.Dense(2048, activation='relu', use_bias=False)(cnn1)

cnn1 = tf.keras.layers.Dropout(0.4)(cnn1)

cnn1 = tf.keras.layers.Dense(1024, activation='relu', use_bias=False)(cnn1)

cnn1 = tf.keras.layers.Dropout(0.3)(cnn1)

cnn1 = tf.keras.layers.Dense(1, activation='tanh', use_bias=True)(cnn1)



cnn2 = tf.keras.applications.resnet_v2.ResNet152V2(input_shape=(256,256,3), include_top=False, weights=None)(input_1)

cnn2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu', use_bias=True)(cnn2)

cnn2 = tf.keras.layers.Flatten()(cnn2)

cnn2 = tf.keras.layers.Dense(2048, activation='relu', use_bias=False)(cnn2)

cnn2 = tf.keras.layers.Dropout(0.4)(cnn2)

cnn2 = tf.keras.layers.Dense(1024, activation='relu', use_bias=False)(cnn2)

cnn2 = tf.keras.layers.Dropout(0.3)(cnn2)

cnn2 = tf.keras.layers.Dense(1, activation='tanh', use_bias=True)(cnn2)



cnn3 = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(256,256,3), include_top=False, weights=None)(input_1)

cnn3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu', use_bias=True)(cnn3)

cnn3 = tf.keras.layers.Flatten()(cnn3)

cnn3 = tf.keras.layers.Dense(2048, activation='relu', use_bias=False)(cnn3)

cnn3 = tf.keras.layers.Dropout(0.4)(cnn3)

cnn3 = tf.keras.layers.Dense(1024, activation='relu', use_bias=False)(cnn3)

cnn3 = tf.keras.layers.Dropout(0.3)(cnn3)

cnn3 = tf.keras.layers.Dense(1, activation='tanh', use_bias=True)(cnn3)



cnn4 = tf.keras.applications.xception.Xception(input_shape=(256,256,3), include_top=False, weights=None)(input_1)

cnn4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), activation='relu', use_bias=True)(cnn4)

cnn4 = tf.keras.layers.Flatten()(cnn4)

cnn4 = tf.keras.layers.Dense(2048, activation='relu', use_bias=False)(cnn4)

cnn4 = tf.keras.layers.Dropout(0.4)(cnn4)

cnn4 = tf.keras.layers.Dense(1024, activation='relu', use_bias=False)(cnn4)

cnn4 = tf.keras.layers.Dropout(0.3)(cnn4)

cnn4 = tf.keras.layers.Dense(1, activation='tanh', use_bias=True)(cnn4)



cnn_output = tf.keras.layers.Concatenate(axis=-1)([cnn1, cnn2, cnn3, cnn4])

cnn_output = tf.keras.layers.Dense(1, activation='relu', use_bias=True)(cnn_output)



x = tf.keras.layers.Concatenate(axis=-1)([cnn_output, input_2, input_3, input_4])

x = tf.keras.layers.Dense(10, activation='relu', use_bias=True)(x)

x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(10, activation='relu', use_bias=True)(x)

x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True)(x)



model = tf.keras.models.Model(inputs=[input_1,input_2,input_3,input_4], outputs=x)



model.summary()

METRICS = [

    tf.keras.metrics.TruePositives(name='tp'),

    tf.keras.metrics.FalsePositives(name='fp'),

    tf.keras.metrics.TrueNegatives(name='tn'),

    tf.keras.metrics.FalseNegatives(name='fn'), 

    tf.keras.metrics.BinaryAccuracy(name='accuracy'),

    tf.keras.metrics.Precision(name='precision'),

    tf.keras.metrics.Recall(name='recall'),

    tf.keras.metrics.AUC(name='auc'),

]

model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(),

              metrics=METRICS)

weight_for_0 = (1 / neg)*(total)/2.0 

weight_for_1 =(1 / pos)*(total)/2.0



class_weight = {0: weight_for_0, 1: weight_for_1}



print('Weight for class 0: {:.2f}'.format(weight_for_0))

print('Weight for class 1: {:.2f}'.format(weight_for_1))

STEPS = np.ceil(total/10)



model.fit(training_dataset, 

          epochs=1, 

          validation_steps=5,

          validation_data = validation_dataset,

          # class_weight=class_weight,

          steps_per_epoch=STEPS)