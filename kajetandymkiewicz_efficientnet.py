import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import matplotlib.pyplot as plt
all_xray_df = pd.read_csv('../input/data/Data_Entry_2017.csv')

#create dict with image number as a key and his absolute path as a value
all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('..', 'input','data','images*', 'images', '*.png'))}

#add path column to data_entry using above dict 
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
from itertools import chain

#get unique class labels (excluding No Finding)
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))

# filtr 'No Finding' label, as we want to have only disease binary class labels
all_labels = [label for label in all_labels if(label != "No Finding")]

for c_label in all_labels:
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
#create column disesase vector - class vector
all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
#split into training and validation dataset
from sklearn.model_selection import train_test_split


#This stratify parameter makes a split so that the proportion of values in the sample produced will be the same
#as the proportion of values provided to parameter stratify.
#For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones,
#stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.

#here we will provide to stratify four labels from Finding Labels column.
#We should check if we couldn't just provide disease_vec
train_df, valid_df = train_test_split(all_xray_df, 
                                   test_size = 0.1,
                                   random_state=2137,
                                   stratify = all_xray_df['Finding Labels'].map(lambda x: x[:4]))

print('train', train_df.shape[0], 'validation', valid_df.shape[0])
#image size to be reconsidered
IMG_W = 256
IMG_H = 256
IMG_SIZE = (IMG_W, IMG_H)
from keras.preprocessing.image import ImageDataGenerator
# #parameters to be checked (samplewise_center,samplewise_std_normalization,shear_range)
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
valid_df['newLabel'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
train_df['newLabel'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)

train_gen = core_idg.flow_from_dataframe(dataframe=train_df, directory=None, x_col = 'path',
            y_col = 'newLabel', class_mode = 'categorical',
            classes = all_labels, target_size = IMG_SIZE, color_mode = 'rgb',
            batch_size = 64)

valid_gen = core_idg.flow_from_dataframe(dataframe=valid_df, directory=None, x_col = 'path',
            y_col = 'newLabel', class_mode = 'categorical',
            classes = all_labels, target_size = IMG_SIZE, color_mode = 'rgb',
            batch_size = 128) # we can use much larger batches for evaluation

# test_X, test_Y = next(core_idg.flow_from_dataframe(dataframe=valid_df, 
#                 directory=None,
#                 x_col = 'path', y_col = 'newLabel', 
#                 class_mode = 'categorical', classes = all_labels,
#                 target_size = IMG_SIZE,
#                 color_mode = 'rgb', batch_size = 4096))
!pip install -U efficientnet
from efficientnet.keras import EfficientNetB1
import tensorflow as tf
# from tf.keras.applications import MobileNetV2
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Activation
from keras.models import Sequential


base_model = EfficientNetB1(input_shape =  (256,256,3), 
                                 include_top = False, weights = None)
multi_disease_model = Sequential()
multi_disease_model.add(base_model)
multi_disease_model.add(GlobalAveragePooling2D())
# multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(128))
multi_disease_model.add(BatchNormalization())
multi_disease_model.add(Activation('relu'))
multi_disease_model.add(Dense(128))
multi_disease_model.add(Activation('relu'))
# multi_disease_model.add(Dropout(0.5))
multi_disease_model.add(Dense(14, activation = 'sigmoid'))

#metrics mae (need to be checked)
multi_disease_model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                           metrics = ['categorical_accuracy'])
multi_disease_model.load_weights('../input/efficient-model/xray_class_weights.best.hdf5')
# pred_Y = multi_disease_model.predict(test_X, batch_size = 32, verbose = True)
from sklearn.metrics import roc_curve, auc

avg = []

fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:,idx].astype(int), pred_Y[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    avg.append(auc(fpr, tpr))
    
print(sum(avg) / len(avg))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('trained_net.png')
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15)
callbacks_list = [checkpoint, early]
multi_disease_model.fit_generator(train_gen, 
                                  steps_per_epoch = 100,
                                  validation_data =  valid_gen,
                                  validation_steps = 20,
                                  epochs = 1, 
                                  callbacks = callbacks_list)