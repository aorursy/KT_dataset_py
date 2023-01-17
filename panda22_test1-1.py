%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob 

import matplotlib.pyplot as plt

import os

import pandas as pd

import seaborn as sns

from skimage.util import montage as montage2d

from skimage.io import imread

base_dir = os.path.join('..', 'input', 'pulmonary-chest-xray-abnormalities')

all_xray_df = pd.read_csv('../input/data/Data_Entry_2017.csv')

all_xray_df.sample(5)
all_xray_df = pd.read_csv('../input/data/Data_Entry_2017.csv')

all_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join('..', 'input', 'data',  'images*', '*', '*.png'))}

print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])

all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)

#all_xray_df['Cardiomegaly'] = all_xray_df['Finding Labels'].map(lambda x: 'Cardiomegaly' in x)

all_xray_df['Patient Age'] = np.clip(all_xray_df['Patient Age'], 5, 100)

all_xray_df['Patient Male'] = all_xray_df['Patient Gender'].map(lambda x: x.upper()=='M').astype('float32')

all_xray_df.sample(3)
label_counts = all_xray_df['Finding Labels'].value_counts()[:15]

fig, ax1 = plt.subplots(1,1,figsize = (12, 8))

ax1.bar(np.arange(len(label_counts))+0.5, label_counts)

ax1.set_xticks(np.arange(len(label_counts))+0.5)

_ = ax1.set_xticklabels(label_counts.index, rotation = 90)
all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))

from itertools import chain

all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))

all_labels = [x for x in all_labels if len(x)>0]

print('All Labels ({}): {}'.format(len(all_labels), all_labels))

for c_label in all_labels:

    if len(c_label)>1: # leave out empty labels

        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

all_xray_df.sample(3)
# keep at least 1000 cases

MIN_CASES = 1000

all_labels = [c_label for c_label in all_labels if all_xray_df[c_label].sum()>MIN_CASES]

print('Clean Labels ({})'.format(len(all_labels)), 

      [(c_label,int(all_xray_df[c_label].sum())) for c_label in all_labels])
# since the dataset is very unbiased, we can resample it to be a more reasonable collection

# weight is 0.1 + number of findings

sample_weights = all_xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2

sample_weights /= sample_weights.sum()

all_xray_df = all_xray_df.sample(40000, weights=sample_weights)



label_counts = all_xray_df['Finding Labels'].value_counts()[:15]

fig, ax1 = plt.subplots(1,1,figsize = (12, 8))

ax1.bar(np.arange(len(label_counts))+0.5, label_counts)

ax1.set_xticks(np.arange(len(label_counts))+0.5)

_ = ax1.set_xticklabels(label_counts.index, rotation = 90)
label_counts = 100*np.mean(all_xray_df[all_labels].values,0)

fig, ax1 = plt.subplots(1,1,figsize = (12, 8))

ax1.bar(np.arange(len(label_counts))+0.5, label_counts)

ax1.set_xticks(np.arange(len(label_counts))+0.5)

ax1.set_xticklabels(all_labels, rotation = 90)

ax1.set_title('Adjusted Frequency of Diseases in Patient Group')

_ = ax1.set_ylabel('Frequency (%)')
all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(all_xray_df, 

                                   test_size = 0.25, 

                                   random_state = 2018,

                                   stratify = all_xray_df['Finding Labels'].map(lambda x: x[:4]))

print('train', train_df.shape[0], 'validation', valid_df.shape[0])
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input

from PIL import Image

IMG_SIZE = (512, 512) # slightly smaller than vgg16 normally expects

core_idg = ImageDataGenerator(samplewise_center=False, 

                              samplewise_std_normalization=False, 

                              horizontal_flip=False, 

                              vertical_flip=False, 

                              height_shift_range=0.1, 

                              width_shift_range=0.1, 

                              brightness_range=[0.7, 1.5],

                              rotation_range=3, 

                              shear_range=0.01,

                              fill_mode='nearest',

                              zoom_range=0.125,

                             preprocessing_function=preprocess_input)
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):

    df_gen = img_data_gen.flow_from_dataframe(in_df,

                                              x_col=path_col,

                                              y_col=y_col,

                                     class_mode = 'raw',

                                    **dflow_args)

    return df_gen
train_gen = flow_from_dataframe(core_idg, train_df, 

                             path_col = 'path',

                            y_col = 'disease_vec', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 8)



valid_gen = flow_from_dataframe(core_idg, valid_df, 

                             path_col = 'path',

                            y_col = 'disease_vec', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 256) # we can use much larger batches for evaluation

# used a fixed dataset for evaluating the algorithm

test_X, test_Y = next(flow_from_dataframe(core_idg, 

                               valid_df, 

                             path_col = 'path',

                            y_col = 'disease_vec', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 400)) # one big batch
t_x, t_y = next(train_gen)

fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))

for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):

    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -127, vmax = 127)

    c_ax.set_title(', '.join([n_class for n_class, n_score in zip(all_labels, c_y) 

                             if n_score>0.5]))

    c_ax.axis('off')
base_pretrained_model = PTModel(input_shape =  t_x.shape[1:], 

                              include_top = False, weights = 'imagenet')

base_pretrained_model.trainable = False
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda, AvgPool2D

from keras.models import Model

pt_features = Input(base_pretrained_model.get_output_shape_at(0)[1:], name = 'feature_input')

pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]

from keras.layers import BatchNormalization

bn_features = BatchNormalization(name='Features_BN')(pt_features)
attn_layer = Conv2D(128, kernel_size = (1,1), padding = 'same', activation = 'elu')(bn_features)

attn_layer = Conv2D(32, kernel_size = (1,1), padding = 'same', activation = 'elu')(attn_layer)

attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'elu')(attn_layer)

attn_layer = AvgPool2D((2,2), strides = (1,1), padding = 'same')(attn_layer) # smooth results

attn_layer = Conv2D(1, 

                    kernel_size = (1,1), 

                    padding = 'valid', 

                    activation = 'softmax',

                   name='AttentionMap2D')(attn_layer)

# fan it out to all of the channels

up_c2_w = np.ones((1, 1, 1, pt_depth))

up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', name='UpscaleAttention',

               activation = 'linear', use_bias = False, weights = [up_c2_w])

up_c2.trainable = False

attn_layer = up_c2(attn_layer)
mask_features = multiply([attn_layer, bn_features])

gap_features = GlobalAveragePooling2D()(mask_features)

gap_mask = GlobalAveragePooling2D()(attn_layer)

# to account for missing values from the attention model

gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
gap_dr = Dropout(0.5)(gap)

dr_steps = Dropout(0.5)(Dense(128, activation = 'elu')(gap_dr))

out_layer = Dense(13, activation = 'softmax')(dr_steps)



attn_model = Model(inputs = [pt_features], outputs = [out_layer], name = 'attention_model')



attn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',

                           metrics = ['binary_accuracy'])



attn_model.summary()
from keras.utils.vis_utils import model_to_dot

from IPython.display import Image

Image(model_to_dot(attn_model, show_shapes=True).create_png())
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('xray_class')



checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)





reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=10) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early, reduceLROnPlat]
from keras.models import Sequential

from keras.optimizers import Adam

tb_model = Sequential(name = 'combined_model')

base_pretrained_model.trainable = False

tb_model.add(base_pretrained_model)

tb_model.add(attn_model)

tb_model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy',

                           metrics = ['binary_accuracy'])

tb_model.summary()
#train_gen.batch_size = 24

tb_model.fit_generator(train_gen, 

                      validation_data = (test_X, test_Y), 

                       steps_per_epoch=train_gen.n//train_gen.batch_size,

                      epochs = 30, 

                      callbacks = callbacks_list,

                      workers = 3)

        
tb_model.load_weights(weight_path)