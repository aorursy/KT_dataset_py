

# -*- encoding: utf-8 -*-

'''



@Author  :   Yukai Song 



'''



# here put the import lib

from datetime import datetime

from logging import log

from os import listdir

from os.path import isfile, join

import librosa

import librosa.display

import numpy as np

import pandas as pd

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.utils import to_categorical



from sklearn.preprocessing import LabelEncoder





import logging



def extract_features(file_name):

    """

    This function takes in the path for an audio file as a string, loads it, and returns the MFCC

    of the audio"""

   

    try:

        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20) 

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        pad_width = max_pad_len - mfccs.shape[1]

        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        

    except Exception as e:

        print("Error encountered while parsing file: ", file_name)

        return None 

     

    return mfccs



def construct_model(oh_labels_shape):

    num_rows = 40

    num_columns = 862

    num_channels = 1



    num_labels = oh_labels_shape

    filter_size = 2

    # Construct model 

    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=filter_size,

                    input_shape=(num_rows, num_columns, num_channels), activation='relu'))

    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.2))



    model.add(Conv2D(filters=32, kernel_size=filter_size, activation='relu'))

    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.2))



    model.add(Conv2D(filters=64, kernel_size=filter_size, activation='relu'))

    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.2))



    model.add(Conv2D(filters=128, kernel_size=filter_size, activation='relu'))

    model.add(MaxPooling2D(pool_size=2))

    model.add(Dropout(0.2))



    model.add(GlobalAveragePooling2D())



    model.add(Dense(num_labels, activation='softmax')) 



    # Compile the model

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 



    # Display model architecture summary 

    model.summary()



    model.load_weights('../input/model-for-using/mymodel2_173.h5')



    return model



#config logging

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别

                    filename='result.log',

                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志

                    #a是追加模式，默认如果不写的话，就是追加模式

                    format=

                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'

                    #日志格式

                    )



mypath = '../input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files'

filenames_total = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))] 



# read file list from 2.txt

file_list = []

with open('../input/voice-config/Voice_config.txt','r') as f:

    for i in f.readlines():

        if str(i[:-1]).isdigit():

            file_list.append(i[:-1])

f.close()



filenames = []

for sfile in filenames_total:

    if sfile[:3] in file_list:

        filenames.append(sfile)

        



# filenames = filenames[:10]





p_id_in_file = [] # patient IDs corresponding to each file

for name in filenames:

    p_id_in_file.append(int(name[:3]))



p_id_in_file = np.array(p_id_in_file) 





max_pad_len = 862 # to make the length of all MFCC equal





filepaths = [join(mypath, f) for f in filenames] # full paths of files



p_diag = pd.read_csv("../input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv",header=None) # patient diagnosis file



labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file]) # labels for audio files

# labels = labels[:40]

features = [] 





# Iterate through each sound file and extract the features

for file_name in filepaths:

    data = extract_features(file_name)

    features.append(data)

print('Finished feature extraction from ', len(features), ' files')



features = np.array(features) # convert to numpy array



# delete the very rare diseases

features1 = np.delete(features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0) 



labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)



# print class counts

unique_elements, counts_elements = np.unique(labels1, return_counts=True)

print(np.asarray((unique_elements, counts_elements)))



# One-hot encode labels

le = LabelEncoder()

# i_labels = le.fit_transform(labels1)

c_label = {'Bronchiectasis':0, 'Bronchiolitis':1, 'COPD':2, 'Healthy':3, 'Pneumonia':4, 'URTI':5}



i_labels = np.array(list(map(lambda x :c_label[x], labels1)))

oh_labels = to_categorical(i_labels,6) 

# add channel dimension for CNN

features1 = np.reshape(features1, (*features1.shape,1)) 



x_test, y_test = features1, oh_labels



model = construct_model(oh_labels.shape[1])



# Calculate pre-training accuracy 

score = model.evaluate(x_test, y_test, verbose=1)

accuracy = 100*score[1]



print("test accuracy: %.4f%%" % accuracy)



preds = model.predict(x_test)

classpreds = np.argmax(preds, axis=1)

y_testclass = np.argmax(y_test, axis=1) # true classes





c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']

for i in range(len(y_testclass)):

    print("语音文件：{2},真实语音来源：{0},预测语音来源：{1}".format(c_names[y_testclass[i]],c_names[classpreds[i]],filenames[i]))

    logging.debug("语音文件：{2},真实语音来源：{0},预测语音来源：{1}".format(c_names[y_testclass[i]],c_names[classpreds[i]],filenames[i]))



print('done')