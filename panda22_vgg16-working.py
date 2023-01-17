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

all_xray_df['Pneumothorax'] = all_xray_df['Finding Labels'].map(lambda x: 'Pneumothorax' in x)

all_xray_df['Patient Age'] = np.clip(all_xray_df['Patient Age'], 5, 100)

all_xray_df['Patient Male'] = all_xray_df['Patient Gender'].map(lambda x: x.upper()=='M').astype('float32')

all_xray_df.sample(3)
sns.pairplot(all_xray_df[['Patient Age', 'Patient Male', 'Pneumothorax']], hue='Pneumothorax')
positive_cases = np.sum(all_xray_df['Pneumothorax']==True)//2

oversample_factor = 4 # maximum number of cases in negative group so it isn't super rare

more_balanced_df = all_xray_df.groupby(['Patient Gender', 'Pneumothorax']).apply(lambda x: x.sample(min(oversample_factor*positive_cases, x.shape[0]), 

                                                                                   replace = False)

                                                      ).reset_index(drop = True)



print(more_balanced_df['Pneumothorax'].value_counts())

sns.pairplot(more_balanced_df[['Patient Age', 'Pneumothorax']], hue='Pneumothorax')
from sklearn.model_selection import train_test_split

raw_train_df, test_valid_df = train_test_split(more_balanced_df, 

                                   test_size = 0.30, 

                                   random_state = 2018,

                                   stratify = more_balanced_df[['Pneumothorax', 'Patient Gender']])

valid_df, test_df = train_test_split(test_valid_df, 

                                   test_size = 0.40, 

                                   random_state = 2018,

                                   stratify = test_valid_df[['Pneumothorax', 'Patient Gender']])

print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0], 'test', test_df.shape[0])

print('train', raw_train_df['Pneumothorax'].value_counts())

print('test', test_df['Pneumothorax'].value_counts())

raw_train_df.sample(1)
train_df = raw_train_df.groupby(['Pneumothorax']).apply(lambda x: x.sample(2000, replace = True)

                                                      ).reset_index(drop = True)

print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
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

                            y_col = 'Pneumothorax', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 8)



valid_gen = flow_from_dataframe(core_idg, valid_df, 

                             path_col = 'path',

                            y_col = 'Pneumothorax', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 256) # we can use much larger batches for evaluation

# used a fixed dataset for evaluating the algorithm

test_X, test_Y = next(flow_from_dataframe(core_idg, 

                               valid_df, 

                             path_col = 'path',

                            y_col = 'Pneumothorax', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 400)) # one big batch

# used a fixed dataset for final evaluation

final_test_X, final_test_Y = next(flow_from_dataframe(core_idg, 

                               test_df, 

                             path_col = 'path',

                            y_col = 'Pneumothorax', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 400)) # one big batch
t_x, t_y = next(train_gen)

fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))

for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):

    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -127, vmax = 127)

    c_ax.set_title('%s' % ('Pneumothorax' if c_y>0.5 else 'Healthy'))

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

                    activation = 'sigmoid',

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

out_layer = Dense(1, activation = 'sigmoid')(dr_steps)



attn_model = Model(inputs = [pt_features], outputs = [out_layer], name = 'attention_model')



attn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',

                           metrics = ['binary_accuracy'])



attn_model.summary()
from keras.utils.vis_utils import model_to_dot

from IPython.display import Image

Image(model_to_dot(attn_model, show_shapes=True).create_png())
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_weights.best.hdf5".format('pneumo_attn')



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
train_gen.batch_size = 24

tb_model.fit_generator(train_gen, 

                      validation_data = (test_X, test_Y), 

                       steps_per_epoch=train_gen.n//train_gen.batch_size,

                      epochs = 3, 

                      callbacks = callbacks_list,

                      workers = 3)
tb_model.load_weights(weight_path)
# get the attention layer since it is the only one with a single output dim

for attn_layer in attn_model.layers:

    c_shape = attn_layer.get_output_shape_at(0)

    if len(c_shape)==4:

        if c_shape[-1]==1:

            print(attn_layer)

            break
import keras.backend as K

rand_idx = np.random.choice(range(len(test_X)), size = 6)

attn_func = K.function(inputs = [attn_model.get_input_at(0), K.learning_phase()],

           outputs = [attn_layer.get_output_at(0)]

          )

fig, m_axs = plt.subplots(len(rand_idx), 2, figsize = (8, 4*len(rand_idx)))

[c_ax.axis('off') for c_ax in m_axs.flatten()]

for c_idx, (img_ax, attn_ax) in zip(rand_idx, m_axs):

    cur_img = test_X[c_idx:(c_idx+1)]

    cur_features = base_pretrained_model.predict(cur_img)

    attn_img = attn_func([cur_features, 0])[0]

    img_ax.imshow(cur_img[0,:,:,0], cmap = 'bone')

    attn_ax.imshow(attn_img[0, :, :, 0], cmap = 'viridis', 

                   vmin = 0, vmax = 1, 

                   interpolation = 'lanczos')

    real_label = test_Y[c_idx]

    img_ax.set_title('Pneumo\nClass:%s' % (real_label))

    pred_confidence = tb_model.predict(cur_img)[0]

    attn_ax.set_title('Attention Map\nPred:%2.1f%%' % (100*pred_confidence[0]))

fig.savefig('attention_map.png', dpi = 300)
pred_Y = tb_model.predict(test_X, 

                          batch_size = 32, 

                          verbose = True)
from sklearn.metrics import classification_report, confusion_matrix

plt.matshow(confusion_matrix(test_Y, pred_Y>0.5))

print(classification_report(test_Y, pred_Y>0.5, target_names = ['Healthy', 'Pneumothorax']))
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(test_Y, pred_Y)

fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 250)

ax1.plot(fpr, tpr, 'b.-', label = 'VGG-Model (AUC:%2.2f)' % roc_auc_score(test_Y, pred_Y))

ax1.plot(fpr, fpr, 'k-', label = 'Random Guessing')

ax1.legend(loc = 4)

ax1.set_xlabel('False Positive Rate')

ax1.set_ylabel('True Positive Rate');

fig.savefig('roc.pdf')
final_pred_Y = tb_model.predict(final_test_X, 

                                verbose = True, 

                                batch_size = 4)
plt.matshow(confusion_matrix(final_test_Y, final_pred_Y>0.5))

print(classification_report(final_test_Y, final_pred_Y>0.5, target_names = ['Healthy', 'Pneumothorax']))
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(final_test_Y, final_pred_Y)

fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 250)

ax1.plot(fpr, tpr, 'b.-', label = 'VGG-Model (AUC:%2.2f)' % roc_auc_score(test_Y, pred_Y))

ax1.plot(fpr, fpr, 'k-', label = 'Random Guessing')

ax1.legend(loc = 4)

ax1.set_xlabel('False Positive Rate')

ax1.set_ylabel('True Positive Rate');

fig.savefig('roc.pdf')
tb_model.save('full_pred_model.h5')
img_in = Input(t_x.shape[1:])

feat_lay = base_pretrained_model(img_in)

just_attn = Model(inputs = attn_model.get_input_at(0), 

      outputs = [attn_layer.get_output_at(0)], name = 'pure_attention')

attn_img = just_attn(feat_lay)

pure_attn_model = Model(inputs = [img_in], outputs = [attn_img], name = 'just_attention_model')

pure_attn_model.save('pure_attn_model.h5')

pure_attn_model.summary()