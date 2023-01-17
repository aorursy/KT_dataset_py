# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import IPython as ipt  #naumually listen and check audio after process 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import glob



from scipy.io import wavfile

import scipy.signal as sps



import matplotlib

import matplotlib.pyplot as plt



import librosa   #down sample the wave 

# Any results you write to the current directory are saved as output.
from keras import losses, models, optimizers

from keras.activations import relu, softmax

from keras.callbacks import (EarlyStopping, LearningRateScheduler,

                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)

from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D, 

                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)

from keras.utils import Sequence, to_categorical
BASEPATH='../input/'

MAX_FRAME=44100*3

MAX_LEN=3

SR=16000

LR=0.0006

EPOCHS=200
#  voice tags representation

# input: original train set

# output: transfer voice tags into sparse matrix

# out of 4790 examples, more than 700 have mutli tags



#find unique tag form list

def sparse_df(train_set,label_list):



    #train_set=train_curated

    #labels_list=[]

    #for label in train_set['labels']:

    #    label=label.split(',')

    #    labels_list.extend(label)

    #labels_set=sorted(list(set(labels_list)))



    #form dict map the unique tags to numbers

    label_dict={}

    label_nr=0

    for label in label_list:

        label_dict[label]=label_nr

        label_nr=label_nr+1



    #transfer tags into sparse matrix

    tags_matrix=np.zeros([len(train_set),len(label_list)])

    for i,label in enumerate(train_set['labels']):

        for label_word in label.split(','):

            tags_matrix[i,label_dict[label_word]]=1

    tags_df=pd.DataFrame(data=tags_matrix,columns=label_dict.keys())



    assert len(tags_df)==len(train_set)



    return pd.concat([train_set,tags_df],axis=1)



#read the train data and re-represent as sparse matrix 

sample_submission=pd.read_csv(BASEPATH+'sample_submission.csv')

train_curated=sparse_df(pd.read_csv(BASEPATH+'train_curated.csv'),sample_submission.columns[1:])
def read_audio0(fname,max_frame):

    file_path=BASEPATH+'/train_curated/'+fname

    _,wave=wavfile.read(file_path)

    w_max=np.max(wave)

    w_min=np.min(wave)

    wave=(wave-w_min)/(w_max-w_min+1e-8)

    if len(wave)>max_frame:

        start_point=np.random.randint(len(wave)-max_frame)

        wave=wave[start_point:start_point+max_frame]

    else:

        wave=np.append(np.zeros(max_frame-len(wave)),wave)

    return wave,file_path
def read_audio(path,fname,max_len,sr):

    file_path=BASEPATH+path+fname

    wave,_=librosa.load(file_path,sr=sr,res_type='kaiser_fast')

    max_frame=max_len*sr

    w_max=np.max(wave)

    w_min=np.min(wave)

    wave=(wave-w_min)/(w_max-w_min+1e-8)

    if len(wave)>max_frame:

        start_point=np.random.randint(len(wave)-max_frame)

        wave=wave[start_point:start_point+max_frame]

    else:

        wave=np.append(np.zeros(max_frame-len(wave)),wave)

    return wave,file_path
#test read_audio 

wave_sample,sample_file=read_audio('train_curated/',train_curated['fname'][6],MAX_LEN,SR)

plt.plot(wave_sample)

ipt.display.Audio(sample_file)
def get_1d_conv_model(input_length,output_length):

    

    nclass = output_length

#   input_length = MAX_FRAME

    

    inp = Input(shape=(input_length,1))

    x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)

    x = Convolution1D(16, 9, activation=relu, padding="valid")(x)

    x = MaxPool1D(16)(x)

    x = Dropout(rate=0.1)(x)

    

    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)

    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)

    x = MaxPool1D(4)(x)

    x = Dropout(rate=0.1)(x)

    

    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)

    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)

    x = MaxPool1D(4)(x)

    x = Dropout(rate=0.1)(x)

    

    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)

    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)

    x = GlobalMaxPool1D()(x)

    x = Dropout(rate=0.2)(x)



    x = Dense(64, activation=relu)(x)

    x = Dense(1028, activation=relu)(x)

    out = Dense(nclass, activation=softmax)(x)



    model = models.Model(inputs=inp, outputs=out)

    opt = optimizers.Adam(LR)    



    model.compile(optimizer=opt,loss=losses.categorical_crossentropy, metrics=['acc'])

    return model
# prepare the train set 

def prepare_train_set(path,df,max_len,sr):

    shape0=len(df)

    train_set=np.zeros((shape0,max_len*SR))

    for i,file in enumerate(df['fname']):

        wave,_=read_audio(path,file,max_len,sr)

        if i%500==0:

            print(i,file)

        train_set[i,:]=wave

    return train_set

    
train_set=prepare_train_set('train_curated/',train_curated,MAX_LEN,SR)

train_set=train_set.reshape(train_set.shape+(1,))

train_set_label=train_curated.drop(['fname','labels'],axis=1).values



    
model=get_1d_conv_model(train_set.shape[1],80)

model.fit(x=train_set,y=train_set_label,epochs=EPOCHS)
##test set preparation

import glob

files=glob.glob(BASEPATH+'test/*.wav')

test_set=np.zeros((len(files),MAX_LEN*SR,1))

test_file=[]

for i,file in enumerate(files):

    file=file[-12:]

    test_file.append(file)

    wave,_=read_audio('test/',file,MAX_LEN,SR)

    test_set[i,:,0]=wave
##predict and submission

submission=model.predict(test_set)

submission_df=pd.DataFrame(data=submission,columns=sample_submission.columns[1:])

submission_df.insert(0,'fname',test_file)

submission_df.to_csv('submission.csv',index=False)
model.summary()