# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
weightpath = 'vvg16-4channels-80epochs-batch64'
datapath = 'human-protein-atlas-image-classification'
import matplotlib.pyplot as plt
%matplotlib inline

import pandas as pd
import numpy as np
import keras
from keras.preprocessing.image import *
from keras.layers import Input,Dense, Dropout, Flatten, Add
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.models import Sequential
from keras.models import Model
from keras import optimizers
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
train_df = pd.read_csv('../input/{}/train.csv'.format(datapath))
#train_df = pd.read_csv('../input/train.csv')
train_df.head()
print(train_df.shape)
print("set size is {}".format(train_df.size))
#the train dataset is reduced in order to save memory
train_df_subset = train_df.sample(frac=0.5, random_state=2)
train_df_subset = train_df_subset.reset_index(drop=True)

print("Subsample subset size is {}".format(train_df_subset.size))
train_df_subset['target_list'] = train_df_subset['Target'].map(lambda x: [int(a) for a in x.split(' ')])
print(train_df_subset.head())

# create a categorical vector
from itertools import chain
from collections import Counter
all_labels = list(chain.from_iterable(train_df_subset['target_list'].values))
c_val = Counter(all_labels)
print(c_val)
n_keys = c_val.keys()
print(n_keys)
max_idx = max(n_keys)
train_df_subset['target_categorial'] = train_df_subset['target_list'].map(lambda ck: [i in ck for i in range(max_idx+1)])
train_df_subset.sample(3)
print(train_df_subset.head())
name_label_dict = {
0:  "Nucleoplasm", 
1:  "Nuclear membrane",   
2:  "Nucleoli",   
3:  "Nucleoli fibrillar center" ,  
4:  "Nuclear speckles"   ,
5:  "Nuclear bodies"   ,
6:  "Endoplasmic reticulum",   
7:  "Golgi apparatus"   ,
8:  "Peroxisomes"   ,
9:  "Endosomes"   ,
10:  "Lysosomes"   ,
11:  "Intermediate filaments",   
12:  "Actin filaments"   ,
13:  "Focal adhesion sites",   
14:  "Microtubules"   ,
15:  "Microtubule ends",   
16:  "Cytokinetic bridge",   
17:  "Mitotic spindle"   ,
18:  "Microtubule organizing center" ,  
19:  "Centrosome"   ,
20:  "Lipid droplets",   
21:  "Plasma membrane",   
22:  "Cell junctions"  , 
23:  "Mitochondria"   ,
24:  "Aggresome"   ,
25:  "Cytosol",
26:  "Cytoplasmic bodies",   
27:  "Rods & rings" 
}
from itertools import chain
from collections import Counter

all_labels = list(chain.from_iterable(train_df_subset['target_list'].values))
c_val = Counter(all_labels)
n_keys = c_val.keys()
max_idx = max(n_keys)
fig, ax1 = plt.subplots(1,1, figsize = (10, 5))
ax1.bar(n_keys, [c_val[k] for k in n_keys])
ax1.set_xticks(range(max_idx+1))
ax1.set_xticklabels([name_label_dict[k] for k in range(max_idx+1)], rotation=90)
for k,v in c_val.items():
    print(name_label_dict[k], 'count:', v)
# out_df_list = []
# for k,v in c_val.items():
#     if v<500:
#         keep_rows = train_df_subset['target_list'].map(lambda x: k in x)
#         out_df_list += [train_df_subset[keep_rows].sample(500, replace=True)]   
# train_df_subset = pd.concat(out_df_list, ignore_index=True)
# print(train_df_subset.shape)
# train_df_subset.head()
print("Subsample subset size is {}".format(train_df_subset.size))
def loadAndStackImageData(df, datatype = 'train', resolution=128):
    colors =  ('green','blue','yellow','red')
    df["images"] = [ ((np.stack( (img_to_array(load_img("../input/{}/{}/{}_{}.png".format(datapath,datatype, idx, color),
                                                               color_mode="grayscale",
                                                               target_size=(resolution,resolution)))
                                                                  for color in colors), axis=-1)).reshape(128,128,4))/255.
                                                                     for idx in tqdm_notebook(df.Id)]
# def loadAndStackImageData(df, datatype = 'train', resolution=128):
#     colors =  ('green','blue','yellow','red')
#     df["images"] = [ ((np.stack( (img_to_array(load_img("../input/{}/{}_{}.png".format(datatype, idx, color),
#                                                                color_mode="grayscale",
#                                                                target_size=(resolution,resolution)))
#                                                                   for color in colors), axis=-1)).reshape(128,128,4))/255.
#                                                                       for idx in tqdm_notebook(df.Id)]
# loadAndStackImageData(train_df_subset,'train', resolution=128)
# #colors =  ('green','blue','yellow','red')
# fig, axs = plt.subplots(4,15,figsize=(15,4))
# for i in range(60):
#     x1 = train_df_subset['images'][i][:,:,0]
#     x2 = train_df_subset['images'][i][:,:,1]
#     x3 = train_df_subset['images'][i][:,:,2]
#     x4 = train_df_subset['images'][i][:,:,3]
#     ax = axs[int(i/15), i % 15]
#     ax.imshow(x1,alpha=0.5)
#     ax.imshow(x2, alpha=0.5)
#     ax.imshow(x3, alpha=0.5)
#     ax.imshow(x4, alpha=0.5)
#     ax.axis('off')
# plt.show()
def vgg16like(num_classes, BATCH_NORM = None):

    input_layer = Input((128, 128, 4))
    
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(input_layer)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)
    
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv4')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv4')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv4')(x)
    x = BatchNormalization()(x)if BATCH_NORM else None
    x = Activation('relu')(x)

    x = Flatten()(x)

    x = Dense(4096)(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(4096, name='fc2')(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(num_classes)(x)
    x = BatchNormalization()(x) if BATCH_NORM else None
    x = Activation('sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=x)

    return model
# from sklearn.model_selection import train_test_split
# train_df, valid_df = train_test_split(train_df_subset, 
#                  test_size = 0.3)
# print(train_df.shape[0], 'training masks')
# print(valid_df.shape[0], 'validation masks')
model = vgg16like(max_idx+1, BATCH_NORM = True)
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.summary()
# x_train = np.array(train_df['images'].tolist())
# y_train = np.array(train_df['target_categorial'].tolist())
# x_valid = np.array(valid_df['images'].tolist())
# y_valid = np.array(valid_df['target_categorial'].tolist())
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# checkpointer = ModelCheckpoint('VGG16_4channels_80epochs_batch64.model', verbose=2, save_best_only=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=2)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=1, factor=0.1)

# history = model.fit(x_train,y_train,batch_size=64,epochs=80,
#                     verbose=1,
#                     callbacks=[checkpointer, early_stopping, reduce_lr],
#                     validation_data=(x_valid,y_valid))
model.load_weights('../input/{}/VGG16_4channels_80epochs_batch64.model'.format(weightpath))
test_df = pd.read_csv('../input/{}/sample_submission.csv'.format(datapath))
loadAndStackImageData(test_df,'test',resolution=128)
x_test = np.array(test_df['images'].tolist())
y_test = model.predict(x_test)
prediction = [ (np.arange(28)[y_test[row]>=0.3]) for row in range(y_test.shape[0])] 
prediction = list(map(lambda x: x.tolist(), prediction))
prediction
#results = [if not idx: ' ' else : ' '.join(str(prt))  for idx in prediction for prt in idx]
results = []
for idx in prediction:
    print(idx)
    if not idx:
        print('.isempty')
        results.append('0')
    else:
        concat = ""
        for elt in idx:
            concat+=str(elt)
            concat+=' '
        concat = concat.strip()
        results.append(concat)
results 
# for idx in prediction: 
#     if not idx:
#         results.append('0')
#     else:
#         res = ""
#         for prt in idx:
#             res+= str(prt)
#         results.append(res)
results
submission = pd.read_csv('../input/{}/sample_submission.csv'.format(datapath))
submission['Predicted'] = results
submission.to_csv('submission.csv', index=False)
