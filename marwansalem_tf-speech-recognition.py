! pip install -q tensorflow-io
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
!pip install py7zr

!ls train/audio

all_labels = ['_background_noise_', 'dog', 'four', 'left', 'off', 'seven', 'three', 'wow', 'bed', 'down', 'go', 'marvin', 'on', 'sheila', 'tree', 'yes', 'bird', 'eight', 'happy', 'nine', 'one', 'six', 'two', 'zero', 'cat', 'five', 'house', 'no', 'right', 'stop', 'up']
classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','unknown']

%cd ..
!python -m py7zr x input/tensorflow-speech-recognition-challenge/train.7z 

#unzip test data
!python -m py7zr x input/tensorflow-speech-recognition-challenge/test.7z 
!ls 
print()
!ls train
#!ls ../input/tensorflow-speech-recognition-challenge/train.7z
#!7z x ./input/tensorflow-speech-recognition-challenge/test.7z -o.
### is this what you wanted ali?? better than linear search i guess
def load_test_file_set(test_file_path='./train/testing_list.txt'):
    #open the txt file containing the pathes of files that should be added to test dataset
    #convert the list to set datastructure
    test_files = open(test_file_path).read().splitlines()
    return set(test_files)

def load_validation_file_set(val_file_path='./train/validation_list.txt'):
    #open the txt file containing the pathes of files that should be added to validation dataset
    #convert the list to set datastructure
    validation_files = open(val_file_path).read().splitlines()
    return set(validation_files)

def is_test_file(test_set, file_path):
    return file_path in test_set

def is_validation_file(val_set, file_path):
    return file_path in val_set



def cut_into_one_sec_segment(audio , sampling_rate):
    length = audio.shape[0]
    #pad audio with zeros

    num_segments = int(np.ceil(length/sampling_rate))
    audio = np.pad(audio, ((0, sampling_rate * num_segments-audio.shape[0]), (0, 0)), 'constant', constant_values=(0, ))
    #print(audio.shape)
    #segments = np.zeros((sampling_rate,1))
    segments = audio[0:  sampling_rate, :]
    #print(segments.shape)
    #print(segments.shape)
    for i in range(1,num_segments):
        seg = audio[ i * sampling_rate : ((i + 1) * sampling_rate ) , :]
        segments = np.hstack((segments, seg))
        #print(len(segments))
    return segments
import os
#outputs  an array of shape(16000, some number)
def load_background_noise(sampling_rate = 16000):##Pads zeros to each background wave file
    #read background music then cut then add to dataset
    bg_noise_dirpath = './train/audio/_background_noise_/'
    segments = None
    first_iter = True
    for filename in os.listdir(bg_noise_dirpath):
        if (filename.endswith('.wav')):
            full_file_path = os.path.join(bg_noise_dirpath, filename)
            audio = tfio.audio.AudioIOTensor(full_file_path).to_tensor().numpy()
            #print(audio.shape)
            
            #first iter return first segment into segments array
            if first_iter == True:
                first_iter = False
                #convert noise audio to 1 second long audio segments with zero padding
                segments = cut_into_one_sec_segment(audio, sampling_rate)
                #print('firstsegshape')
                #print(segments.shape)
            else:#2nd+ iter stack the new segments
                #convert noise audio to 1 second long audio segments with zero padding
                new_segments = cut_into_one_sec_segment(audio, sampling_rate)
                segments = np.hstack((segments, new_segments))
                #print('stackedsegments shape')
                #print(segments.shape)
    #print(segments.shape)
    
    return segments

background_dataset = load_background_noise()
print(background_dataset.shape)
#CELL TO RUN to show output
## this the cell to USe to load the train data

# ALI ZEYAD RUN THIS


# ! mv train/audio/_background_noise_ .
###classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','unknown']
num_wav_files_with_bgnoise = 64727 #(computed before using a forloop)
#number of wav files not counting the background noise clips that have to broken down into smaller units
#FS is 16000 for all inputs

import os

X_train = []
Y_train = []

X_val   = []
Y_val   = []
X_test  = []
Y_test  = []

def read_train_audio(folder_path='./train/audio'):
    
    global X_train, X_val, Y_train, Y_val

    classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go','unknown']
    
    all_labels = ['_background_noise_', 'dog', 'four', 'left', 'off', 'seven', 'three', 'wow', 'bed', 'down', 'go', 'marvin', 'on', 'sheila', 'tree', 'yes', 'bird', 'eight', 'happy', 'nine', 'one', 'six', 'two', 'zero', 'cat', 'five', 'house', 'no', 'right', 'stop', 'up']
    label_id = {}
    for i in range(len(all_labels)):
        label_id[all_labels[i]] = i
        
    class_id = {}
    for i in range(len(classes)):
        class_id[classes[i]] = i
    test_file_set = load_test_file_set()
    validation_file_set = load_validation_file_set()
    
    for label in os.listdir(folder_path): #loop on all folders in the audio folder
        dir_path = os.path.join(folder_path, label)
        if os.path.isfile(dir_path):#skip any files
            continue
        for filename in os.listdir(dir_path): #loop on all files in folder
            if label == '_background_noise_':
                #background noise is loaded in a seperate
                break
            else:
                
                full_file_path = os.path.join(dir_path, filename)
                if os.path.isdir(full_file_path):
                    continue
                #padded_audio = np.zeros(shape=(16000, 1))
                audio = tfio.audio.AudioIOTensor(full_file_path).to_tensor().numpy()

                # size_audio = audio.shape[0]
                # padded_audio[0 : size_audio] = audio
                file_rel_path = os.path.join(label, filename)
                #print(file_rel_path)
                if is_validation_file(validation_file_set, file_rel_path):
                    X_val.append(np.pad(audio, ((0, 16000-audio.shape[0]), (0, 0)), 'constant', constant_values=(0, )))
                    #Y_val.append(class_id.get(label, len(classes) - 1))
                    
                    Y_val.append(label_id.get(label, len(all_labels) - 1))
                elif is_test_file(test_file_set, file_rel_path):
                    #add to test data ??
                    pass
                else:
                    #print(audio.shape)
                    #print(full_file_path)
                    #print(label)
                    X_train.append(np.pad(audio, ((0, 16000-audio.shape[0]), (0, 0)), 'constant', constant_values=(0, )))
                    #Y_train.append(class_id.get(label, len(classes) - 1))
                    
                    Y_train.append(label_id.get(label, len(all_labels) - 1))


    
   
    X_train = np.array(X_train)
    #print(X_train.shape)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    Y_val = Y_val.reshape(Y_val.shape[0], 1)


    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1])

    bgnoise_train_data = load_background_noise()
    bgnoise_train_data = bgnoise_train_data.T
    print(bgnoise_train_data.shape)
    #y_bgnoise = [class_id['unknown']] * bgnoise_train_data.shape[0]
    y_bgnoise = [label_id['_background_noise_']] * bgnoise_train_data.shape[0]
    Y_train.extend(y_bgnoise)
    Y_train = np.array(Y_train)
    Y_train = Y_train.reshape(Y_train.shape[0], 1)

    X_train = np.vstack((X_train, bgnoise_train_data))
    print(X_train.shape)


read_train_audio()
print('X Train (words + background noise) Dimensions:', X_train.shape)
print('Y Train' , Y_train.shape)

print('X Validation' , X_val.shape)
print('Y Validation' , Y_val.shape)

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, Dense
from tensorflow.keras.layers import Bidirectional, LSTM, TimeDistributed


def get_SR_Model(num_classes: int):
    X_input = Input(shape=(16000, 1))
    X = Conv1D(filters=256,kernel_size=15,strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Dropout(0.2)(X)
    X = Conv1D(filters=512,kernel_size=15,strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Dropout(0.2)(X)
    X = LSTM(units=512, return_sequences=True)(X)
    X = LSTM(units=512, return_sequences=False)(X)
    X = Dense(num_classes, activation='softmax')(X)
    return Model(inputs=[X_input], outputs=[X])

model = get_SR_Model(11)
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
#ana bahbed
model.compile(loss=losses.sparse_categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,name='Adam'), metrics=metrics.SparseCategoricalCrossentropy(
    name='sparse_categorical_crossentropy'))
print(model)
## ana bahbed
history = model.fit(X_train, Y_train, batch_size=64, epochs=1)

print(history.history)
print(len(test_file_set))
print(len(validation_file_set))
folder_path='./train/audio'
val_count = 0
test_count = 0
for label in os.listdir(folder_path):
    labelpath = os.path.join(folder_path, label)
    if os.path.isfile(labelpath):
        continue
    for filename in os.listdir(labelpath):
        filepath = os.path.join(label,filename)
        if is_validation_file(validation_file_set, filepath):
            val_count += 1
        elif is_test_file(test_file_set, filepath):
            test_count += 1

print('Validation:' + str(val_count))
print('Test: ' +str(test_count))
print(test_file_set)
read_train_audio_in_dir('train/audio')
X_train.shape