# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

from utils import *

import array 



from pydub import AudioSegment

import numba 

import regex as re

from glob import glob

import numpy as np

import pandas as pd



import tensorflow as tf



from keras.models import Model, Sequential

from keras.layers import Input, Conv2D, Flatten, MaxPooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, concatenate, Dense, Dropout

from keras.optimizers import Adam

from tensorflow.python.keras.utils import to_categorical

from keras_tqdm import TQDMNotebookCallback

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



%matplotlib inline



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
all_labels = [x[0].split('/')[-1] for x in os.walk("../input/train/audio/")]

 

exclusions = ["","_background_noise_"]

POSSIBLE_LABELS = [item for item in all_labels if item not in exclusions]


# POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()

id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}

name2id = {name: i for i, name in id2name.items()}

len(id2name)
all_labels
def load_data(data_dir):

    np.random.seed = 1

    

    """ Return 2 lists of tuples:

    [(class_id, user_id, path), ...] for train

    [(class_id, user_id, path), ...] for validation

    """

    # Just a simple regexp for paths with three groups:

    # prefix, label, user_id

#     pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")

    pattern  =  re.compile("(.+[\/\\\\])?(\w+)[\/\\\\]([^_]+)_.+wav")

    all_files = glob(os.path.join(data_dir, '../input/train/audio/*/*wav'))



    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:

        validation_files = fin.readlines()

        

    valset = set()

    for entry in validation_files:

        r = re.match(pattern, entry)

        if r:

            valset.add(r.group(3))

    

    possible = set(POSSIBLE_LABELS)

    

    train, val, silent, unknown = [], [],[],[]

    

    for entry in all_files:

        r = re.match(pattern, entry)

        if r:

            label, uid = r.group(2), r.group(3)

            

            if label == '_background_noise_': #we've already split up noise files into 1 seg chunks under 'silence' folder

                continue

                

#             if label not in possible:

#                 label = 'unknown'



            label_id = name2id[label]

            sample = (label, label_id, uid, entry)

        

            

            if label == "unknown":

                unknown.append(sample)

            elif label == "silence":

                silent.append(sample)

                

            elif uid in valset:    

                val.append(sample)

            else:

                train.append(sample)



    print('There are {} train and {} val samples'.format(len(train), len(val)))

    

    columns_list = ['label', 'label_id', 'user_id', 'wav_file']

    



    train_df = pd.DataFrame(train, columns = columns_list)

    valid_df = pd.DataFrame(val, columns = columns_list)

    silent_df = pd.DataFrame(silent, columns = columns_list)

    unknown_df = pd.DataFrame(unknown, columns = columns_list)

    

    return train_df, valid_df, unknown_df, silent_df
train_df, valid_df, unknown_df, silent_df = load_data('../input/')
train_df.head()
train_df.shape
train_df['label'].value_counts()
int(valid_df.shape[0]*0.1)
unknown_df.shape,silent_df.shape
#augment validation set with silence and unknown files, made with step=250 when generating silence files

extra_data_size = int(valid_df.shape[0]*0.1)



unknown_val = unknown_df.sample(extra_data_size,random_state=2)

unknown_df = unknown_df[~unknown_df.index.isin(unknown_val.index.values)]



silent_val = silent_df.sample(extra_data_size,random_state=2)

silent_df = silent_df[~silent_df.index.isin(silent_val.index.values)]





valid_df = pd.concat([valid_df,silent_val,unknown_val],axis=0)
silence_files_AS = [AudioSegment.from_wav(x) for x in silent_df.wav_file.values]
import random

random.choice(silence_files_AS)