# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import os

print(os.listdir("../input/omniglot/"))



# Any results you write to the current directory are saved as output.
language_directory_list=[]

character_list=[]

image_name_list=[]

path_list=[]

basic_path="../input/omniglot/images_background/"

for Language_Directory in os.listdir(basic_path):

    for character in os.listdir(basic_path+Language_Directory):

        for image_name in os.listdir(basic_path+Language_Directory+"/"+character):

            path=basic_path+Language_Directory+"/"+character+"/"+image_name

            image_name_list.append(image_name)

            character_list.append(character)

            language_directory_list.append(Language_Directory)

            path_list.append(path)

            
image_name_list = pd.Series(image_name_list)

character_list= pd.Series(character_list)

language_directory_list = pd.Series( language_directory_list)

path_list = pd.Series(  path_list)

df=pd.DataFrame()

df["language"]=language_directory_list

df["character"]=character_list

df["image_name"]=image_name_list

df["path"]= path_list 

df.head(5)
images_list=[]

f, axarr = plt.subplots(5,4, figsize=(10,10))

#you can select you images range from here image range should be twenty in between o to 19000

begning_of_images=1000

ending_of_images=1020

for k in range (begning_of_images,ending_of_images):

    

    

    

    img = cv2.imread(df.path.iloc[k])

    images_list.append(img)

len(images_list)

count=begning_of_images

for i in range(5):

    for j in range(4):

        

            axarr[i,j].title.set_text(df.language.iloc[count])

            axarr[i,j].imshow(images_list.pop())

            count=count+1
language_directory_list=[]

character_list=[]

image_name_list=[]

path_list=[]

basic_path="../input/omniglot/images_evaluation/"

for Language_Directory in os.listdir(basic_path):

    for character in os.listdir(basic_path+Language_Directory):

        for image_name in os.listdir(basic_path+Language_Directory+"/"+character):

            path=basic_path+Language_Directory+"/"+character+"/"+image_name

            image_name_list.append(image_name)

            character_list.append(character)

            language_directory_list.append(Language_Directory)

            path_list.append(path)
image_name_list = pd.Series(image_name_list)

character_list= pd.Series(character_list)

language_directory_list = pd.Series( language_directory_list)

path_list = pd.Series(  path_list)

Evaluation_df=pd.DataFrame()

Evaluation_df["language"]=language_directory_list

Evaluation_df["character"]=character_list

Evaluation_df["image_name"]=image_name_list

Evaluation_df["path"]= path_list 

Evaluation_df.head(5)
images_list=[]

f, axarr = plt.subplots(5,4, figsize=(10,10))

#you can select you images range from here image range should be twenty in between o to 19000

begning_of_images=1000

ending_of_images=1020

for k in range (begning_of_images,ending_of_images):

    

    

    

    img = cv2.imread(Evaluation_df.path.iloc[k])

    images_list.append(img)

len(images_list)

count=begning_of_images

for i in range(5):

    for j in range(4):

        

            axarr[i,j].title.set_text(df.language.iloc[count])

            axarr[i,j].imshow(images_list.pop())

            count=count+1
#here we are adding value again every object in which true indicates that image is present in directory and false represent 

#that image is not in the directory

df['exists'] = df['path'].map(os.path.exists)

Evaluation_df['exists'] = Evaluation_df['path'].map(os.path.exists)

print("Training Dataframe Length",len(df))

print("testing Dataframe length",len(Evaluation_df))

print("\n\n\n")



print("Training Dataframe Length where images = true",len(df[df.exists == True]))

print("Testing Dataframe Length where images = true",len(Evaluation_df[Evaluation_df.exists == True]))

#ok all the images are present in the directory



X_train=[]

y_train=[]



for i in range (0,len(df)):

    img=plt.imread(df.path.iloc[i])

    X_train.append(img)

    y_train.append(df.language.iloc[i])



X_train=np.asarray(X_train)

y_train=np.asarray(y_train)

print("X Train shape",X_train.shape)

print("Y test shape",y_train.shape)

X_test=[]

y_test=[]

for i in range (0,len(Evaluation_df)):

    img=plt.imread(Evaluation_df.path.iloc[i])

    X_test.append(img)

    y_test.append(Evaluation_df.language.iloc[i])



X_test=np.asarray(X_test)

y_test=np.asarray(y_test)

print("Y test shape",y_test.shape)

print("X_test shape",X_test.shape)
X_train=X_train.reshape(X_train.shape[0],105,105,1)

X_test=X_test.reshape(X_test.shape[0],105,105,1)

train_groups = [X_train[np.where(y_train==i)[0]] for i in np.unique(y_train)]

test_groups = [X_test[np.where(y_test==i)[0]] for i in np.unique(y_test)]

print('train groups:', [X.shape[0] for X in train_groups])

print('test groups:', [X.shape[0] for X in test_groups])
w=train_groups[3][23][:,:,0]

plt.imshow(w)

print(train_groups[3][23].shape)

print(w.shape)
# Import Keras and other Deep Learning dependencies

from keras.models import Sequential

import time

from keras.optimizers import Adam

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate

from keras.models import Model

import seaborn as sns

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from keras.layers.merge import Concatenate

from keras.layers.core import Lambda, Flatten, Dense

from keras.initializers import glorot_uniform

from sklearn.preprocessing import LabelBinarizer

from keras.optimizers import *

from keras.engine.topology import Layer

from keras import backend as K

from keras.regularizers import l2

K.set_image_data_format('channels_last')

import cv2

import os

from skimage import io

import numpy as np

from numpy import genfromtxt

import pandas as pd

import tensorflow as tf



import numpy.random as rng

from sklearn.utils import shuffle



%matplotlib inline

%load_ext autoreload

%reload_ext autoreload

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

def gen_random_batch(in_groups, batch_halfsize = 8):

    out_img_a, out_img_b, out_score = [], [], []

    all_groups = list(range(len(in_groups)))

    for match_group in [True, False]:

        group_idx = np.random.choice(all_groups, size = batch_halfsize)

        out_img_a += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in group_idx]

        if match_group:

            b_group_idx = group_idx

            out_score += [1]*batch_halfsize

        else:

            # anything but the same group

            non_group_idx = [np.random.choice([i for i in all_groups if i!=c_idx]) for c_idx in group_idx] 

            b_group_idx = non_group_idx

            out_score += [0]*batch_halfsize

            

        out_img_b += [in_groups[c_idx][np.random.choice(range(in_groups[c_idx].shape[0]))] for c_idx in b_group_idx]

            

    return np.stack(out_img_a,0), np.stack(out_img_b,0), np.stack(out_score,0)
print("Training set language",len(df.language.value_counts()))

print("Testing set language",len(Evaluation_df.language.value_counts()))
def generate_one_hot_encoding(classes):

    encoder = LabelBinarizer()

    transfomed_labels = encoder.fit_transform(classes)

    return transfomed_labels

language=df.language

Evaluation_language=Evaluation_df.language

labels = generate_one_hot_encoding(language)

Evaluation_labels=generate_one_hot_encoding(Evaluation_language)



print("length of labels",len(labels))

print("length of Evaluation_labels",len(Evaluation_labels))
X_train=X_train.reshape(X_train.shape[0],105,105,1)

X_test=X_test.reshape(X_test.shape[0],105,105,1)

pv_a, pv_b, pv_sim = gen_random_batch(train_groups, 3)

fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize = (12, 6))

for c_a, c_b, c_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, m_axs.T):

    ax1.imshow(c_a[:,:,0])

    ax1.set_title('Image A')

    ax1.axis('off')

    ax2.imshow(c_b[:,:,0])

    ax2.set_title('Image B\n Similarity: %3.0f%%' % (100*c_d))

    ax2.axis('off')
from keras.models import Model

from keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout

img_in = Input(shape = X_train.shape[1:], name = 'FeatureNet_ImageInput')

n_layer = img_in

for i in range(2):

    n_layer = Conv2D(8*2**i, kernel_size = (3,3), activation = 'relu')(n_layer)

    n_layer = BatchNormalization()(n_layer)

    n_layer = Activation('relu')(n_layer)

    n_layer = Conv2D(16*2**i, kernel_size = (3,3), activation = 'relu')(n_layer)

    n_layer = BatchNormalization()(n_layer)

    n_layer = Activation('relu')(n_layer)

    n_layer = MaxPool2D((2,2))(n_layer)

n_layer = Flatten()(n_layer)

n_layer = Dense(32, activation = 'linear')(n_layer)

n_layer = Dropout(0.5)(n_layer)

n_layer = BatchNormalization()(n_layer)

n_layer = Activation('relu')(n_layer)

feature_model = Model(inputs = [img_in], outputs = [n_layer], name = 'FeatureGenerationModel')

feature_model.summary()
from keras.layers import concatenate

img_a_in = Input(shape = X_train.shape[1:], name = 'ImageA_Input')

img_b_in = Input(shape = X_train.shape[1:], name = 'ImageB_Input')

img_a_feat = feature_model(img_a_in)

img_b_feat = feature_model(img_b_in)

combined_features = concatenate([img_a_feat, img_b_feat], name = 'merge_features')

combined_features = Dense(64, activation = 'relu')(combined_features)

combined_features = BatchNormalization()(combined_features)

combined_features = Activation('relu')(combined_features)

combined_features = Dense(32, activation = 'relu')(combined_features)

combined_features = BatchNormalization()(combined_features)

combined_features = Activation('relu')(combined_features)

combined_features = Dense(1, activation = 'sigmoid')(combined_features)

similarity_model = Model(inputs = [img_a_in, img_b_in], outputs = [combined_features], name = 'Similarity_Model')

similarity_model.summary()

    
# setup the optimization process

similarity_model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

def show_model_output(nb_examples = 3):

    pv_a, pv_b, pv_sim = gen_random_batch(test_groups, nb_examples)

    pred_sim = similarity_model.predict([pv_a, pv_b])

    fig, m_axs = plt.subplots(2, pv_a.shape[0], figsize = (12, 6))

    for c_a, c_b, c_d, p_d, (ax1, ax2) in zip(pv_a, pv_b, pv_sim, pred_sim, m_axs.T):

        ax1.imshow(c_a[:,:,0])

        ax1.set_title('Image A\n Actual: %3.0f%%' % (100*c_d))

        ax1.axis('off')

        ax2.imshow(c_b[:,:,0])

        ax2.set_title('Image B\n Predicted: %3.0f%%' % (100*p_d))

        ax2.axis('off')

    return fig

# a completely untrained model

_ = show_model_output()
# make a generator out of the data

def siam_gen(in_groups, batch_size = 32):

    while True:

        pv_a, pv_b, pv_sim = gen_random_batch(train_groups, batch_size//2)

        yield [pv_a, pv_b], pv_sim

# we want a constant validation group to have a frame of reference for model performance

valid_a, valid_b, valid_sim = gen_random_batch(test_groups, 1024)

loss_history = similarity_model.fit_generator(siam_gen(train_groups), 

                               steps_per_epoch = 500,

                               validation_data=([valid_a, valid_b], valid_sim),

                                              epochs = 50,

                                             verbose = True)
_ = show_model_output()
t_shirt_vec = np.stack([train_groups[0][0]]*X_test.shape[0],0)

t_shirt_score = similarity_model.predict([t_shirt_vec, X_test], verbose = True, batch_size = 128)

ankle_boot_vec = np.stack([train_groups[-1][0]]*X_test.shape[0],0)

ankle_boot_score = similarity_model.predict([ankle_boot_vec, X_test], verbose = True, batch_size = 128)