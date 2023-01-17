# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm.auto import tqdm
from glob import glob
import time, gc
import cv2

from tensorflow import keras
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import clone_model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
from matplotlib import pyplot as plt
import seaborn as sns
from keras.utils import plot_model
%matplotlib inline
gc.enable()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
test_df = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')
class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')
sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
train_df.head()
test_df
sample_sub_df.head()
class_map_df.head()
print('Size of training data: {}'.format(train_df.shape))
print('Size of test data: {}'.format(test_df.shape))
print('Size of class map: {}'.format(class_map_df.shape))
count = class_map_df.groupby('component_type').count().reset_index()
count
# plotting labels. belonged to each class 
sns.barplot('component_type','label',data = count)
 # now visualising some of the top labels in each class
# uses pd.groupby to count the occurence of each label in all three classes    
root_component = train_df.groupby('grapheme_root').count().reset_index()[['grapheme_root','image_id']].sort_values(by = 'image_id', ascending = False)
vowel_component = train_df.groupby('vowel_diacritic').count().reset_index().sort_values(by = 'image_id', ascending = False)[['vowel_diacritic','image_id']]
cons_component = train_df.groupby('consonant_diacritic').count().reset_index().sort_values(by = 'image_id', ascending = False)[['consonant_diacritic','image_id']]
root_component
# images vs grapheme_root
plt.figure(figsize = (10,10))
sns.barplot('grapheme_root','image_id',data = root_component,order = root_component['grapheme_root'] )
# images vs grapheme_root
plt.figure(figsize = (10,10))
sns.barplot('grapheme_root','image_id',data = root_component,order = root_component['grapheme_root'] )#
# plotting 10 most occuring among them
plt.figure(figsize = (10,10))
sns.barplot('grapheme_root','image_id',data = root_component.iloc[0:10],order = root_component['grapheme_root'][0:10] )
# now finding how these 10 looks like
np.array(root_component['grapheme_root'].map(dict(class_map_df[class_map_df['component_type']=='grapheme_root'][['label', 'component']].values)))[0:10]
# plotting 10 least among given
plt.figure(figsize = (10,10))
sns.barplot('grapheme_root','image_id',data = root_component.iloc[-10:],order = root_component['grapheme_root'][-10:] )
# now finding how these 10 looks like
np.array(root_component['grapheme_root'].map(dict(class_map_df[class_map_df['component_type']=='grapheme_root'][['label', 'component']].values)))[-10:]
# same for vowel_diacritics
# images vs vowel_diacritic
plt.figure(figsize = (10,10))
sns.barplot('vowel_diacritic','image_id',data = vowel_component,order = vowel_component['vowel_diacritic'] )
# now finding how these  looks like
np.array(vowel_component['vowel_diacritic'].map(dict(class_map_df[class_map_df['component_type']=='vowel_diacritic'][['label', 'component']].values)))
# same for 'consonant_diacritic'
# images vs 'consonant_diacritic'
plt.figure(figsize = (10,10))
sns.barplot('consonant_diacritic','image_id',data = cons_component,order = cons_component['consonant_diacritic'] )
# now finding how these  looks like
np.array(cons_component['consonant_diacritic'].map(dict(class_map_df[class_map_df['component_type']=='consonant_diacritic'][['label', 'component']].values)))
train_df = train_df.drop(['grapheme'], axis=1, inplace=False)
train_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')
train_df
IMG_SIZE=64
N_CHANNELS=1
def resize(df, size=64, need_progress_bar=True):
    resized = {}
    resize_size=64
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):
            image=df.loc[df.index[i]].values.reshape(137,236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0 
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
                xmin = min(ls_xmin)
                ymin = min(ls_ymin)
                xmax = max(ls_xmax)
                ymax = max(ls_ymax)
                roi = image[ymin:ymax,xmin:xmax]
                resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
                resized[df.index[i]] = resized_roi.reshape(-1)

    else:
        for i in range(df.shape[0]):
            #image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            image=df.loc[df.index[i]].values.reshape(137,236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0 
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
                xmin = min(ls_xmin)
                ymin = min(ls_ymin)
                xmax = max(ls_xmax)
                ymax = max(ls_ymax)

                roi = image[ymin:ymax,xmin:xmax]
                resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
                resized[df.index[i]] = resized_roi.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized
def get_dummies(df):
    cols = []
    for col in df:
        cols.append(pd.get_dummies(df[col].astype(str)))
    return pd.concat(cols, axis=1)
# model preparation

inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 1))

model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1))(inputs)
model = Conv2D(filters=32, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = Dropout(rate=0.25)(model)

model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Dropout(rate=0.25)(model)

model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=128, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = Dropout(rate=0.2)(model)

model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.15)(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Conv2D(filters=256, kernel_size=(5, 5), padding='SAME', activation='relu')(model)
model = BatchNormalization(momentum=0.20)(model)
model = Dropout(rate=0.25)(model)

model = Flatten()(model)
model = Dense(512, activation = "relu",name= 'dense_')(model)
model = Dropout(rate=0.25)(model)
dense = Dense(256, activation = "relu",name= 'dense_1')(model)

head_root = Dense(168, activation = 'softmax',name= 'dense_2')(dense)
head_vowel = Dense(11, activation = 'softmax',name= 'dense_3')(dense)
head_consonant = Dense(7, activation = 'softmax',name= 'dense_4')(dense)

model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant]) # 3 outputs one for each
model.summary()
# plotting the model how it exactly look like
plot_model(model, to_file='model.png')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Set a learning rate annealer. Learning rate will be half after 3 epochs if accuracy is not increased
learning_rate_reduction_root = ReduceLROnPlateau(monitor='dense_2_accuracy', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)
learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='dense_3_accuracy', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)
learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='dense_4_accuracy', 
                                            patience=3, 
                                            verbose=1,
                                            factor=0.5, 
                                            min_lr=0.00001)
batch_size = 256
epochs = 16
# creating a custom data augmentor which supports multi class output

class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):

    def flow(self,
             x,
             y=None,
             batch_size=256,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir= None,
             save_prefix='',
             save_format='png',
             subset=None):

        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)
            
        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,
                                     shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:, i: i + target_length]
                i += target_length

            yield flowx, target_dict
HEIGHT = 137
WIDTH = 236
gc.collect()
# training iteratively using loops

gc.enable()
histories = []
for i in range(3):
    
    gc.collect()
    len(gc.get_objects())
    train_df_new = pd.merge(pd.read_parquet(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet'), train_df.iloc[i*len(train_df)//4:(i+1)*len(train_df)//4], on='image_id').drop(['image_id'], axis=1)
    
    # Visualize few samples of current training dataset
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))
    count=0
    for row in ax:
        for col in row:
            col.imshow(resize(train_df_new.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]], need_progress_bar=False).values.reshape(-1).reshape(IMG_SIZE, IMG_SIZE).astype(np.float64))
            count += 1
    plt.show()
    
    X_train = train_df_new.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)
    X_train = resize(X_train)/255
    
    # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images
    X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)
    
    Y_train_root = pd.get_dummies(train_df_new['grapheme_root']).values
    Y_train_vowel = pd.get_dummies(train_df_new['vowel_diacritic']).values
    Y_train_consonant = pd.get_dummies(train_df_new['consonant_diacritic']).values

    print(f'Training images: {X_train.shape}')
    print(f'Training labels root: {Y_train_root.shape}')
    print(f'Training labels vowel: {Y_train_vowel.shape}')
    print(f'Training labels consonants: {Y_train_consonant.shape}')

    # Divide the data into training and validation set
    x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)
    gc.collect()
    
    del train_df_new
    del X_train
    del Y_train_root
    del Y_train_vowel
    del Y_train_consonant
    len(gc.get_objects())
    gc.collect()

    # Data augmentation for creating more training data
    datagen = MultiOutputDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.15, # Randomly zoom image 
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
        
    # This will just calculate parameters required to augment the given data. This won't perform any augmentations
    datagen.flow(x_train)

    # Fit the model
    history = model.fit_generator(datagen.flow(x_train, {'dense_2': y_train_root, 'dense_3': y_train_vowel, 'dense_4': y_train_consonant}, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]), 
                              steps_per_epoch=x_train.shape[0] // batch_size, 
                              callbacks=[learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant])

    histories.append(history)
    
    del datagen
    gc.collect()
    len(gc.get_objects())
    # Delete to reduce memory usage
    del x_train
    del x_test
    del y_train_root
    del y_test_root
    del y_train_vowel
    del y_test_vowel
    del y_train_consonant
    del y_test_consonant
   
    gc.collect()
    
    
gc.collect()
# functions for plotting losses and accuracies

def plot_loss(his, epoch, title):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')
    plt.plot(np.arange(0, epoch), his.history['dense_2_loss'], label='train_root_loss')
    plt.plot(np.arange(0, epoch), his.history['dense_3_loss'], label='train_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['dense_4_loss'], label='train_consonant_loss')
    
    plt.plot(np.arange(0, epoch), his.history['val_dense_2_loss'], label='val_train_root_loss')
    plt.plot(np.arange(0, epoch), his.history['val_dense_3_loss'], label='val_train_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['val_dense_4_loss'], label='val_train_consonant_loss')
    
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def plot_acc(his, epoch, title):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['dense_2_accuracy'], label='train_root_acc')
    plt.plot(np.arange(0, epoch), his.history['dense_3_accuracy'], label='train_vowel_accuracy')
    plt.plot(np.arange(0, epoch), his.history['dense_4_accuracy'], label='train_consonant_accuracy')
    
    plt.plot(np.arange(0, epoch), his.history['val_dense_2_accuracy'], label='val_root_acc')
    plt.plot(np.arange(0, epoch), his.history['val_dense_3_accuracy'], label='val_vowel_accuracy')
    plt.plot(np.arange(0, epoch), his.history['val_dense_4_accuracy'], label='val_consonant_accuracy')
    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.show()
# plotting graphs for all four sets
for i in range(3):
    plot_loss(histories[i], epochs, f'Training Dataset: {i}')
    plot_acc(histories[i], epochs, f'Training Dataset: {i}')
    
del histories
gc.collect()
len(gc.get_objects())
model.save('bengalimodal.h5')


