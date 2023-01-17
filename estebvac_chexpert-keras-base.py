# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import matplotlib.pyplot as plt

import datetime

from pathlib import Path



from sklearn.metrics import roc_auc_score



import numpy as np

import os

import cv2

import warnings

warnings.filterwarnings("ignore")



# IMPORT KERAS LIBRARY

from keras.applications.inception_v3 import InceptionV3

from keras.applications.densenet import DenseNet121

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, Input, GlobalAveragePooling2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

from keras.callbacks import ModelCheckpoint

from keras import backend as K

import tensorflow as tf

import matplotlib.pyplot as plt



%load_ext autoreload

%autoreload
model_path='.'

path='../input/chexpert'

train_folder=f'{path}train'

test_folder=f'{path}test'

train_lbl=f'{path}train_labels.csv'
chestxrays_root = Path(path)

data_path = chestxrays_root/'chexp'
!ls '../input'
full_train_df = pd.read_csv(data_path/'CheXpert-v1.0-small/train.csv')

full_valid_df = pd.read_csv(data_path/'CheXpert-v1.0-small/valid.csv')
chexnet_targets = ['No Finding',

       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',

       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',

       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',

       'Support Devices']



chexpert_targets = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
full_train_df.head()
u_one_features = ['Atelectasis', 'Edema']

u_zero_features = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']
def feature_string(row):

    feature_list = []

    for feature in u_one_features:

        if row[feature] in [-1,1]:

            feature_list.append(feature)

            

    for feature in u_zero_features:

        if row[feature] == 1:

            feature_list.append(feature)

            

    return ';'.join(feature_list)

            

     
full_train_df['train_valid'] = False

full_valid_df['train_valid'] = True
full_train_df['patient'] = full_train_df.Path.str.split('/',3,True)[2]

full_train_df  ['study'] = full_train_df.Path.str.split('/',4,True)[3]



full_valid_df['patient'] = full_valid_df.Path.str.split('/',3,True)[2]

full_valid_df  ['study'] = full_valid_df.Path.str.split('/',4,True)[3]
full_df = pd.concat([full_train_df, full_valid_df])

full_df.head()
full_df['feature_string'] = full_df.apply(feature_string,axis = 1).fillna('')

full_df['feature_string'] =full_df['feature_string'] .apply(lambda x:x.split(";"))

full_df.head()
#get the first 5 whale images

paths =  full_df.Path[:5]

labels = full_df.feature_string[:5]



fig, m_axs = plt.subplots(1, len(labels), figsize = (20, 10))

#show the images and label them

for ii, c_ax in enumerate(m_axs):

    c_ax.imshow(cv2.imread(os.path.join(data_path,paths[ii])))

    c_ax.set_title(labels[ii])
from collections import Counter



labels_count = Counter(label for chexpert_targets in full_df['feature_string'] for label in chexpert_targets)

#plt.bar(chexpert_targets, labels_count.values(), align='center', alpha=0.5)

#plt.show

x_pos = np.arange(len(labels_count.values()))

#Plot the data:

my_colors = 'rgbkymc'

lbls = list.copy(chexpert_targets)

lbls.insert(0,'')

plt.bar(x_pos, labels_count.values(), align='center', alpha=0.5 , color=my_colors)

plt.xticks(x_pos, lbls, rotation='vertical')

sample_perc = 0.00

train_only_df = full_df[~full_df.train_valid]

valid_only_df = full_df[full_df.train_valid]

unique_patients = train_only_df.patient.unique()

mask = np.random.rand(len(unique_patients)) <= sample_perc

sample_patients = unique_patients[mask]



dev_df = train_only_df[full_train_df.patient.isin(sample_patients)]

train_df = train_only_df[~full_train_df.patient.isin(sample_patients)]



print(valid_only_df.Path.size)

print(train_df.Path.size)
datagen=image.ImageDataGenerator(rescale=1./255, 

                                 featurewise_center=True,

                                 featurewise_std_normalization=True,

                                 rotation_range=5,

                                 width_shift_range=0.2,

                                 height_shift_range=0.2,

                                 horizontal_flip=True,

                                 validation_split = 0.1)

test_datagen=image.ImageDataGenerator(rescale=1./255)
def generate_datasets(image_size = 224):



    train_generator=datagen.flow_from_dataframe(dataframe=train_df, directory=data_path, 

                                                x_col="Path", y_col="feature_string", has_ext=True, seed = 42, #classes = chexpert_targets,

                                                class_mode="categorical", target_size=(image_size,image_size), batch_size=32, subset = "training")



    validation_generator = datagen.flow_from_dataframe(dataframe=train_df, directory=data_path, 

                                                       x_col="Path", y_col="feature_string", has_ext=True, seed = 42, #classes = chexpert_targets,

                                                       class_mode="categorical", target_size=(image_size,image_size), batch_size=32, subset = "validation")



    test_generator = test_datagen.flow_from_dataframe(dataframe=valid_only_df, directory=data_path, 

                                                      target_size=(image_size,image_size),class_mode='categorical',

                                                      batch_size=1, shuffle=False, #classes = chexpert_targets,

                                                      x_col="Path", y_col="feature_string")

    

    return [train_generator,validation_generator,test_generator]
def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
def build_model(image_size = 224, load_previous_weights = True, freeze_cnn = False):

    base_model = DenseNet121(include_top= False, input_shape=(image_size,image_size,3), weights='imagenet')



    # add a global spatial average pooling layer

    x = base_model.output

    x = GlobalAveragePooling2D(input_shape=(1024,1,1))(x)

    # Add a flattern layer 

    x = Dense(2048, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    # Add a fully-connected layer

    x = Dense(512, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)

    # and a logistic layer --  we have 5 classes

    predictions = Dense(6, activation='sigmoid')(x)

    

    # this is the model we will train

    model = Model(inputs=base_model.input, outputs=predictions)



    # Recover previously trained weights

    if load_previous_weights:

        try:

            model.load_weights('../input/chexpert-keras-base/weights.hdf5')

            print('Weights successfuly loaded')

        except:

            print('Weights not loaded')



    # first: train only the top layers (which were randomly initialized)

    # i.e. freeze all convolutional InceptionV3 layers

    if freeze_cnn:

        for layer in base_model.layers:

            layer.trainable = False

        

    # compile the model (should be done *after* setting layers to non-trainable)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', auc])

        

    return model
def train_model(model , datasets, epochs=1, image_size = 224):

    

    checkpointer = ModelCheckpoint(filepath='weights.hdf5', 

                                   verbose=1, save_best_only=True)

    

    train_generator,validation_generator,test_generator = datasets

    

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

    STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

    print(STEP_SIZE_TRAIN)

    print(STEP_SIZE_VALID)



    history = model.fit_generator(generator=train_generator,

                        steps_per_epoch=STEP_SIZE_TRAIN,

                        validation_data=validation_generator,

                        validation_steps=STEP_SIZE_VALID,

                        epochs=epochs, callbacks = [checkpointer])

    return history
image_size_input = 224

model = build_model(image_size = image_size_input)
datasets = generate_datasets(image_size = image_size_input)

train_generator,validation_generator,test_generator = datasets
history = train_model(model , datasets, epochs=3, image_size = image_size_input)

history.history
plt.figure(1)    

# summarize history for accuracy     

plt.subplot(211)  

plt.plot(history.history['acc'])  

plt.plot(history.history['val_acc'])  

plt.title('model accuracy')  

plt.ylabel('accuracy')  

plt.xlabel('epoch')  

plt.legend(['train', 'test'], loc='upper left')  



# summarize history for loss    

plt.subplot(212)  

plt.plot(history.history['loss'])  

plt.plot(history.history['val_loss'])  

plt.title('model loss')  

plt.ylabel('loss')  

plt.xlabel('epoch')  

plt.legend(['train', 'test'], loc='upper left')  

plt.show()  
import pickle

with open('/trainHistoryDict', 'wb') as file_pi:

    pickle.dump(history.history, file_pi)
from sklearn.preprocessing import MultiLabelBinarizer

test = pd.Series(test_generator.labels)

mlb = MultiLabelBinarizer()

y_labels = mlb.fit_transform(test)
test_generator.reset()

y_pred_keras = model.predict_generator(test_generator,verbose = 1,steps=test_generator.n)
from sklearn.metrics import roc_curve

from sklearn.metrics import auc



plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')



for ii in range(1, y_pred_keras.shape[1]):

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_labels[:,ii], y_pred_keras[:,ii])

    auc_keras = auc(fpr_keras, tpr_keras)

    plt.plot(fpr_keras, tpr_keras, label=chexpert_targets[ii-1] + '(area = {:.3f})'.format(auc_keras))

    

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()

    



# get the first image of the testing dataset

x = test_generator[0][0]

preds = model.predict(x)
preds = y_pred_keras[1,:]

class_idx = np.argmax(preds)

class_output = model.output[:, class_idx]

#import the last convolutional layer of the model, this depends on the model

last_conv_layer = model.get_layer("conv5_block16_concat")
grads = K.gradients(class_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(1024):

    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# Generate the heatmap

heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)

heatmap /= np.max(heatmap)
img = x[0] 

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = np.uint8(255 * heatmap)

img = np.uint8(255 * img)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

plt.subplot(1,2,1)

plt.imshow(img)

plt.title("Original")

plt.subplot(1,2,2)

plt.imshow(superimposed_img)

plt.title("Heat map")