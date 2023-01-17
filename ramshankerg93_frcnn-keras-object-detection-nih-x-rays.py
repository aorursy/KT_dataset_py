# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import division

from __future__ import print_function

from __future__ import absolute_import

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import os

from glob import glob

%matplotlib inline

import matplotlib.pyplot as plt



import random

import pprint

import sys

import time

import numpy as np

from optparse import OptionParser

import pickle

import math

import cv2

import copy

from matplotlib import pyplot as plt

import tensorflow as tf

import pandas as pd

import os



from sklearn.metrics import average_precision_score



from keras import backend as K

from keras.optimizers import Adam, SGD, RMSprop

from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed

from keras.engine.topology import get_source_inputs

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.objectives import categorical_crossentropy



from keras.models import Model

from keras.utils import generic_utils

from keras.engine import Layer, InputSpec

from keras import initializers, regularizers

all_xray_df = pd.read_csv('/kaggle/input/data/Data_Entry_2017.csv')

all_xray_df_copy = all_xray_df.copy()

all_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join('..', 'input','data', 'images_*', '*', '*.png'))}

#print(all_image_paths)

print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])

all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

#all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x[:-1]))

all_xray_df.sample(3)
all_xray_df.drop("Unnamed: 11",axis=1,inplace=True)
all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

from itertools import chain



all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))

all_labels = [x for x in all_labels if len(x)>0]

print('All Labels ({}): {}'.format(len(all_labels), all_labels))

for c_label in all_labels:

    if len(c_label)>1: # leave out empty labels

        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

all_xray_df.sample(3)
all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values],1).map(lambda x: x[0])
df_annotations = pd.read_csv('../input/data/BBox_List_2017.csv')



print(df_annotations.shape)

df_annotations.drop(['Unnamed: 6','Unnamed: 7','Unnamed: 8'], axis = 1, inplace= True)

df_annotations.head()
new_df = pd.merge(all_xray_df_copy ,df_annotations, on= 'Image Index')

new_df.shape
new_df.head()
all_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join('..', 'input/data', 'images*', '*', '*.png'))}

print('Scans found:', len(all_image_paths), ', Total Headers', new_df.shape[0])

new_df['path'] = new_df['Image Index'].map(all_image_paths.get)

#all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x[:-1]))

new_df


def splitDataFrameList(df,target_column,separator):

    ''' df = dataframe to split,

    target_column = the column containing the values to split

    separator = the symbol used to perform the split

    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 

    The values in the other columns are duplicated across the newly divided rows.

    '''

    def splitListToRows(row,row_accumulator,target_column,separator):

        split_row = row[target_column].split(separator)

        for s in split_row:

            new_row = row.to_dict()

            new_row[target_column] = s

            row_accumulator.append(new_row)

    new_rows = []

    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))

    new_df = pd.DataFrame(new_rows)

    return new_df
stack_df = splitDataFrameList(new_df,"Finding Labels",'|')

stack_df
def read_and_process_image(stack_df):

    """

    Returns two arrays: 

        X is an array of resized images

        y is an array of labels

    """

    X = [] # images

    y = [] # labels

    nrows = 150

    ncolumns = 150

    

    for i,image in enumerate(stack_df["path"]):

        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))  #Read the image

        #get the labels

        label = stack_df.loc[i,"Finding Label"]

        y.append(label)

    

    return np.array(X), np.array(y)
X,y = read_and_process_image(stack_df)

y = pd.get_dummies(y)
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(all_xray_df, 

                                   test_size = 0.25, 

                                   random_state = 2018,

                                   stratify = all_xray_df['Finding Labels'].map(lambda x: x[:4]))

print('train', train_df.shape[0], 'validation', valid_df.shape[0])
from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (150, 150)

core_idg = ImageDataGenerator(samplewise_center=True, 

                              samplewise_std_normalization=True, 

                              horizontal_flip = True, 

                              vertical_flip = False, 

                              height_shift_range= 0.05, 

                              width_shift_range=0.1, 

                              rotation_range=5, 

                              shear_range = 0.1,

                              fill_mode = 'reflect',

                              zoom_range=0.15)
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):

    base_dir = os.path.dirname(in_df[path_col].values[0])

    print('## Ignore next message from keras, values are replaced anyways')

    df_gen = img_data_gen.flow_from_directory(base_dir, 

                                     class_mode = 'sparse',

                                    **dflow_args)

    df_gen.filenames = in_df[path_col].values

    df_gen.classes = np.stack(in_df[y_col].values)

    df_gen.samples = in_df.shape[0]

    df_gen.n = in_df.shape[0]

    df_gen._set_index_array()

    df_gen.directory = '' # since we have the full path

    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))

    return df_gen
train_gen = flow_from_dataframe(core_idg, train_df, 

                             path_col = 'path',

                            y_col = 'disease_vec', 

                            target_size = IMG_SIZE,

                             color_mode = 'grayscale',

                            batch_size = 30)



valid_gen = flow_from_dataframe(core_idg, valid_df, 

                             path_col = 'path',

                            y_col = 'disease_vec', 

                            target_size = IMG_SIZE,

                             color_mode = 'grayscale',

                            batch_size = 256)

# we can use much larger batches for evaluation

# used a fixed dataset for evaluating the algorithm

test_X, test_Y = next(flow_from_dataframe(core_idg, 

                               valid_df, 

                             path_col = 'path',

                            y_col = 'disease_vec', 

                            target_size = IMG_SIZE,

                             color_mode = 'grayscale',

                            batch_size = 1024)) # one big batch
print(valid_gen.n)
import os

os.getcwd()
import pandas as pd

df =  pd.read_csv('../input/data/BBox_List_2017.csv') #,header=None)

df.drop(["Unnamed: 6","Unnamed: 7","Unnamed: 8"],axis=1,inplace=True)

df["Image Index"] = stack_df["path"]

df.columns = [0,1,2,3,4,5]

df = df[[0,2,3,4,5,1]]

df.columns = [0,1,2,3,4,5]

df.head()
df.to_csv('/kaggle/working/BBox_List_2017.csv',index=False,header=False)
import csv

csv_file = '/kaggle/working/BBox_List_2017.csv'

txt_file = '/kaggle/working/BBox_List_2017.txt'

with open(txt_file, "w") as my_output_file:

    with open(csv_file, "r") as my_input_file:

        [ my_output_file.write(",".join(row)+'\n') for row in csv.reader(my_input_file)]

    my_output_file.close()
!python3 /kaggle/working/keras-frcnn/train_frcnn.py -o simple -p BBox_List_2017.txt --hf --vf --rot --input_weight_path resnet50_weights_tf_dim_ordering_tf_kernels.h5 
!git clone https://github.com/kbardool/keras-frcnn.git