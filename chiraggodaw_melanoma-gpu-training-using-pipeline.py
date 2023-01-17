# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import cv2

import gc

from keras.utils import to_categorical



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def preprocessing(dataset):



    dataset['sex'].fillna("no sex", inplace = True)

    dataset['age_approx'].fillna(0, inplace = True)

    dataset['anatom_site_general_challenge'].fillna("NA", inplace = True)

    dataset = dataset.replace(to_replace = ['male'], value = 0)

    dataset = dataset.replace(to_replace = ['female'], value = 1)

    dataset = dataset.replace(to_replace = ['no sex'], value = 2)

    dataset = dataset.replace(to_replace = ['torso'], value = 0)

    dataset = dataset.replace(to_replace = ['lower extremity'], value = 1)

    dataset = dataset.replace(to_replace = ['upper extremity'], value = 2)

    dataset = dataset.replace(to_replace = ['head/neck'], value = 3)

    dataset = dataset.replace(to_replace = ['NA'], value = 4)

    dataset = dataset.replace(to_replace = ['palms/soles'], value = 5)

    dataset = dataset.replace(to_replace = ['oral/genital'], value = 6)

    

    return dataset
df_test = preprocessing(pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv'))

df_test
IMAGE_HEIGHT = 300

IMAGE_WIDTH = 300

TOTAL_SAMPLES = 1440
BATCH_SIZE = 8

SHUFFLE_SIZE = TOTAL_SAMPLES

STEPS_PER_EPOCH = int(TOTAL_SAMPLES/BATCH_SIZE)
def _parse_function_train(proto):

    # define your tfrecord again. Remember that you saved your image as a string.

    keys_to_features = {'image_raw': tf.io.FixedLenFeature([], tf.string),

                        'target': tf.io.FixedLenFeature([], tf.int64),

                        'sex': tf.io.FixedLenFeature([], tf.int64),

                        'age_approx': tf.io.FixedLenFeature([], tf.float32),

                       'anatom_site_general_challenge': tf.io.FixedLenFeature([],tf.int64)}

    

    # Load one example

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    

    image_shape = tf.stack([IMAGE_HEIGHT,IMAGE_WIDTH,3])



    parsed_features['image_raw'] = tf.io.decode_jpeg(parsed_features['image_raw'], 3)#, fixed_length = 256*256*3)

    

    parsed_features['image_raw'] = tf.reshape(parsed_features['image_raw'], image_shape)

        

    parsed_features['image_raw'] = tf.image.random_flip_left_right(parsed_features['image_raw'])

    

    parsed_features['image_raw'] = tf.image.random_flip_up_down(parsed_features['image_raw'])

    

    parsed_features['image_raw'] = tf.image.adjust_saturation(parsed_features['image_raw'],4)

        

    return parsed_features['image_raw'], parsed_features['sex'], parsed_features['age_approx'], parsed_features['anatom_site_general_challenge'],parsed_features['target']
def dataset_fetch (filenames, isTrain):

    

    dataset = tf.data.TFRecordDataset(filenames)

    

    if(isTrain == True):

        dataset = dataset.repeat()



    dataset = dataset.map(_parse_function_train)

        

    dataset = dataset.shuffle(SHUFFLE_SIZE)



    dataset = dataset.batch(BATCH_SIZE)

        

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

            

    return dataset
training_dataset = dataset_fetch('../input/melanoma-tfrecord/train(2).tfrecords',True)

training_dataset
validation_dataset = dataset_fetch('../input/melanoma-classification-eda/validation.tfrecords',True)

validation_dataset
def fetch_data(iterator):

    while True:

        image,sex,age,site,target = iterator.get_next()

        target = to_categorical(target)

        yield ([image,age],target)
def fetch_validation_data(val_iterator):

    while True:

        image,sex,age,site,target = val_iterator.get_next()

        target = to_categorical(target)

        yield ([image,age],target)
iterator_train = iter(training_dataset)

training_dataset_final = fetch_data(iterator_train)



print(training_dataset_final)
iterator_val = iter(validation_dataset)

validation_dataset_final = fetch_validation_data(iterator_val)



print(validation_dataset_final)
image_input = tf.keras.layers.Input(shape = (IMAGE_HEIGHT,IMAGE_WIDTH,3))

x1 = tf.keras.applications.Xception(weights = 'imagenet', include_top = False)(image_input)

x1 = tf.keras.layers.MaxPooling2D((2,2))(x1)

x1 = tf.keras.layers.Flatten()(x1)

x1 = tf.keras.layers.Dense(128, activation = 'relu')(x1)

x1 = tf.keras.layers.Dropout(0.2)(x1)

# x1 = tf.keras.layers.Dense(128, activation = 'relu')(x1)

# x1 = tf.keras.layers.Dropout(0.2)(x1)

image = tf.keras.layers.Dense(32, activation = 'relu',kernel_regularizer = tf.keras.regularizers.l2())(x1)





age_input = tf.keras.layers.Input(shape = (1))

# x2 = tf.keras.layers.Dense(128,activation = 'relu')(age_input)

x2 = tf.keras.layers.Dense(64, activation = 'relu')(age_input)

x2 = tf.keras.layers.Dropout(0.2)(x2)

age = tf.keras.layers.Dense(32, activation = 'relu',kernel_regularizer = tf.keras.regularizers.l2())(x2)



# gender_input = tf.keras.layers.Input(shape = (1))

# # x3 = tf.keras.layers.Dense(128,activation = 'relu')(gender_input)

# x3 = tf.keras.layers.Dense(64, activation = 'relu')(gender_input)

# x3 = tf.keras.layers.Dropout(0.2)(x3)

# gender = tf.keras.layers.Dense(32, activation = 'relu',kernel_regularizer = tf.keras.regularizers.l2())(x3)



# site_input = tf.keras.layers.Input(shape = (1))

# # x4 = tf.keras.layers.Dense(128,activation = 'relu')(site_input)

# x4 = tf.keras.layers.Dense(64, activation = 'relu')(site_input)

# x4 = tf.keras.layers.Dropout(0.2)(x4)

# site = tf.keras.layers.Dense(32, activation = 'relu',kernel_regularizer = tf.keras.regularizers.l2())(x4)



merge1 = tf.keras.layers.concatenate([image,age])



op = tf.keras.layers.Dense(64, activation = 'relu')(merge1)

op = tf.keras.layers.Dropout(0.3)(op)

# op = tf.keras.layers.Dense(64, activation = 'relu')(op)

op = tf.keras.layers.Dense(16, activation = 'relu',kernel_regularizer = tf.keras.regularizers.l2())(op)

output_final = tf.keras.layers.Dense(2, activation = 'softmax')(op)



model = tf.keras.models.Model(inputs = [image_input,age_input], outputs = output_final)
model.compile( optimizer=tf.keras.optimizers.Adamax(),

    loss='binary_crossentropy',

    metrics=['accuracy'],)
model.summary()
def build_lrfn(lr_start=0.00001, lr_max=0.0001, 

               lr_min=0.000001, lr_rampup_epochs=20, 

               lr_sustain_epochs=0, lr_exp_decay=.8):

    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn



lrfn = build_lrfn()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
model.fit(training_dataset_final,

          epochs = 50, 

          #validation_data = validation_dataset_final,validation_steps = 150,

         steps_per_epoch = STEPS_PER_EPOCH)

gc.collect()
predictions_list = np.array([], dtype = 'float32')

print(predictions_list)

for row in df_test.iterrows():

    im_test = cv2.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/test/' + row[1]['image_name'] + '.jpg')

    im_resize = np.reshape(cv2.resize(im_test, (IMAGE_HEIGHT, IMAGE_WIDTH)), (1,IMAGE_HEIGHT,IMAGE_WIDTH,3))

    predictions_list = np.append(predictions_list,

                                 

                                 model.predict([im_resize,

#                                                 np.reshape(row[1]['sex'],(1)),

                                                np.reshape(row[1]['age_approx'],(1)),

#                                                 np.reshape(row[1]['anatom_site_general_challenge'],(1)),

                                               ]

                                              )[0][1])



    print(predictions_list.shape, end = "\r")
gc.collect()
unique_elements, counts_elements = np.unique(predictions_list, return_counts=True)



print(np.asarray((unique_elements, counts_elements)))
test_image_name = df_test['image_name'].to_numpy()
sample_submission = pd.DataFrame({"image_name":test_image_name, "target":predictions_list})

sample_submission
sample_submission.to_csv("submission.csv",index = False)