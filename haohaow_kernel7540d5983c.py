# import packages

from datetime import datetime

import os

import librosa

import librosa.display

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import re

import tensorflow as tf

import numpy as np

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets



# AUTO = tf.data.experimental.AUTOTUNE





# Detect TPU, and gives appropriate distribution strategy

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
# define a instruct class to save constants

class CONFIG:

    ROOT = "/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/"

    FILE_PATH = os.path.join(ROOT, 'audio_and_txt_files')

    

    MFCC_NUM = 40

    PADDING = 862 # to make the length of all MFCC equal

    EPOCHES = [50, 250, 100, 50]

    BATCHSIZE = [2, 128, 2, 64]

    
file_names = [file_name for file_name in os.listdir(CONFIG.FILE_PATH) if '.wav' in file_name] 

file_paths = [os.path.join(CONFIG.FILE_PATH, file_name) for file_name in file_names]

patient_id = []



for name in file_names:

    patient_id.append(name.split('_')[0])

    

patient_id = np.array(patient_id)
def extract_features(file_name, n_mfcc=CONFIG.MFCC_NUM, pad_width = CONFIG.PADDING):

    '''

    param:

        file_name: os.path.join(CONFIG.FILE_PATH, file_name)

        n_mfcc: the row of feature tensor, corresponds to the number of time segmentation

        pad_width: to make sure the shapes for all tensors are same

    return:

        mfccs: a tensor with shape[n_mfcc, pad_width, 1]

    '''

    try:

        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20) 

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        pad_width = pad_width - mfccs.shape[1]

        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        

    except Exception as e:

        print("Error encountered while parsing file: ", file_name)

        return None 

     

    return mfccs
label_df = pd.read_csv(os.path.join(CONFIG.ROOT, 'patient_diagnosis.csv'), header=None, names = ['id', 'label'])# dtype = {'id': int}

print(label_df.head(2))



labels = [label_df[label_df['id'] == int(x)]['label'].values for x in patient_id] # int 保证数据类型一致，否则返回[]

print('labels[0]={}'.format(labels[0]))
features = [] 



# Iterate through each sound file and extract the features

for file_path in file_paths:

    data = extract_features(file_path)

    features.append(data)



print('Finished feature extraction from ', len(features), ' files')

features = np.array(features)
print('{} slices（batches）, {} frequencies, {} time point'.format(features.shape[0], features.shape[1], features.shape[2]))



plt.figure(figsize=(10, 4))

librosa.display.specshow(features[7], x_axis='ms',y_axis='mel')

plt.colorbar()

plt.title('MFCC')

plt.tight_layout()

plt.show()
labels = np.array(labels)



features_cleaned = np.delete(features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0) 

labels_cleaned = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)



# Results Reviewing

unique_elements, counts_elements = np.unique(labels_cleaned, return_counts=True)

print(np.asarray((unique_elements, counts_elements)))

print('There are {} slices（batches）, {} frequencies, {} time point'.format(features_cleaned.shape[0], features_cleaned.shape[1], features_cleaned.shape[2]))



le = LabelEncoder()

i_labels = le.fit_transform(labels_cleaned)

onehot_labels = to_categorical(i_labels) 

print(onehot_labels.shape)

features2 = np.reshape(features_cleaned, (*features_cleaned.shape,1)) 

print(features2.shape)
x_train, x_test, y_train, y_test = train_test_split(features2, onehot_labels, stratify=onehot_labels,test_size=0.302, random_state = 42)
num_rows = 40

num_columns = 862

num_channels = 1

from keras.regularizers import l2 

num_labels = onehot_labels.shape[1]

filter_size = 2 # it would be better to use 3, but the input image is too small.

with strategy.scope():

    # Construct model 

    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=filter_size, input_shape=(num_rows, num_columns, num_channels), activation='relu', kernel_regularizer=l2(0.001)))

    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.3))



    model.add(Conv2D(filters=32, kernel_size=filter_size, activation='relu', kernel_regularizer=l2(0.001)))

    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.3))



    model.add(Conv2D(filters=64, kernel_size=filter_size, activation='relu', kernel_regularizer=l2(0.001)))

    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.2))



    model.add(Conv2D(filters=128, kernel_size=filter_size, activation='relu', kernel_regularizer=l2(0.001)))

    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.2))



    model.add(GlobalAveragePooling2D())



    model.add(Dense(num_labels, activation='softmax')) 

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 

    model.summary()



    score = model.evaluate(x_test, y_test, verbose=1)

    accuracy = 100*score[1]



    print("Pre-training accuracy: %.4f%%" % accuracy)
# seperate training and defining process. This will cause that we can train our model for many epoches without making params zero.

with strategy.scope():



    callbacks = [

        ModelCheckpoint(

            filepath='mymodel2_{epoch:02d}.h5',

            save_best_only=True,

            monitor='val_accuracy',

            verbose=1)

    ]

    start = datetime.now()



#     stats1 = model.fit(x_train, y_train, batch_size=CONFIG.BATCHSIZE[0], epochs=CONFIG.EPOCHES[0],

#               validation_data=(x_test, y_test), callbacks=callbacks, verbose=1)

    



#     stats2 = model.fit(x_train, y_train, batch_size=CONFIG.BATCHSIZE[1], epochs=CONFIG.EPOCHES[1],

#               validation_data=(x_test, y_test), callbacks=callbacks, verbose=1)

    

#     stats3 = model.fit(x_train, y_train, batch_size=CONFIG.BATCHSIZE[2], epochs=CONFIG.EPOCHES[2],

#               validation_data=(x_test, y_test), callbacks=callbacks, verbose=1)

    

    stats4 = model.fit(x_train, y_train, batch_size=CONFIG.BATCHSIZE[3], epochs=CONFIG.EPOCHES[3],

              validation_data=(x_test, y_test), callbacks=callbacks, verbose=1)



    duration = datetime.now() - start

    print("Training completed in time: ", duration)
# Evaluating the model on the training and testing set

score = model.evaluate(x_train, y_train, verbose=0)

print("Training Accuracy: ", score[1])



score = model.evaluate(x_test, y_test, verbose=0)

print("Testing Accuracy: ", score[1])