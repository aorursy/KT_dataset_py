import gc

import os

import json

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

from PIL import Image

import matplotlib.pyplot as plt

import keras

from keras import layers

from keras.applications import DenseNet121

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.models import Sequential

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')
IS_LOCAL = False

if(IS_LOCAL):

    PATH="../input/iwildcam/"

else:

    PATH="../input/iwildcam-2019-fgvc6/"

os.listdir(PATH)
%%time

train_df = pd.read_csv(os.path.join(PATH, 'train.csv'))

test_df = pd.read_csv(os.path.join(PATH, 'test.csv'))
x_train = np.load('../input/reducing-image-sizes-to-32x32/X_train.npy')

x_test = np.load('../input/reducing-image-sizes-to-32x32/X_test.npy')

y_train = np.load('../input/reducing-image-sizes-to-32x32/y_train.npy')
train_df.head()
test_df.head()
print("Train and test shape: {} {}".format(train_df.shape, test_df.shape))
classes_wild = {0: 'empty', 1: 'deer', 2: 'moose', 3: 'squirrel', 4: 'rodent', 5: 'small_mammal', \

                6: 'elk', 7: 'pronghorn_antelope', 8: 'rabbit', 9: 'bighorn_sheep', 10: 'fox', 11: 'coyote', \

                12: 'black_bear', 13: 'raccoon', 14: 'skunk', 15: 'wolf', 16: 'bobcat', 17: 'cat',\

                18: 'dog', 19: 'opossum', 20: 'bison', 21: 'mountain_goat', 22: 'mountain_lion'}
train_df['classes_wild'] = train_df['category_id'].apply(lambda cw: classes_wild[cw])
train_df.head()
train_image_files = list(os.listdir(os.path.join(PATH,'train_images')))

test_image_files = list(os.listdir(os.path.join(PATH,'test_images')))

                         

print("Number of image files: train:{} test:{}".format(len(train_image_files), len(test_image_files)))
%%time

train_file_names = list(train_df['file_name'])

print("Matching train image names: {}".format(len(set(train_file_names).intersection(train_image_files))))
%%time

test_file_names = list(test_df['file_name'])

print("Matching test image names: {}".format(len(set(test_file_names).intersection(test_image_files))))
cnt_classes_images = train_df.classes_wild.nunique()

print("There are {} classes of images".format(cnt_classes_images))

pd.DataFrame(train_df.classes_wild.value_counts()).transpose()
def plot_classes(feature, fs=8, show_percents=True, color_palette='Set3'):

    f, ax = plt.subplots(1,1, figsize=(2*fs,4))

    total = float(len(train_df))

    g = sns.countplot(train_df[feature], order = train_df[feature].value_counts().index, palette=color_palette)

    g.set_title("Number and percentage of labels for each class of {}".format(feature))

    if(show_percents):

        for p in ax.patches:

            height = p.get_height()

            ax.text(p.get_x()+p.get_width()/2.,

                    height + 3,

                    '{:1.2f}%'.format(100*height/total),

                    ha="center") 

    plt.show()    
plot_classes('classes_wild')
plot_classes('seq_num_frames', fs=3)
plot_classes('location', fs=15)
fig, ax = plt.subplots(1,1,figsize=(16,26))

t = pd.DataFrame(train_df.groupby(['classes_wild', 'location'])['seq_id'].count().reset_index())

m = t.pivot(index='location', columns='classes_wild', values='seq_id')

s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu")

s.set_title('Number of wild animals observed per location', size=16)

plt.show()
fig, ax = plt.subplots(1,1,figsize=(16,26))

tmp = train_df[train_df['classes_wild'] != 'empty']

t = pd.DataFrame(tmp.groupby(['classes_wild', 'location'])['seq_id'].count().reset_index())

m = t.pivot(index='location', columns='classes_wild', values='seq_id')

s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu")

s.set_title('Number of wild animals observed per location (except `empty`)', size=16)

plt.show()

del t, tmp, m

gc.collect()
plot_classes('rights_holder', fs=3)
fig, ax = plt.subplots(1,1,figsize=(16,4))

t = pd.DataFrame(train_df.groupby(['classes_wild', 'rights_holder'])['seq_id'].count().reset_index())

m = t.pivot(index='rights_holder', columns='classes_wild', values='seq_id')

s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu")

s.set_title('Number of wild animals observed by each rights holder', size=16)

plt.show()
fig, ax = plt.subplots(1,1,figsize=(16,4))

t = pd.DataFrame(train_df[~(train_df.classes_wild == 'empty')].groupby(['classes_wild', 'rights_holder'])['seq_id'].count().reset_index())

m = t.pivot(index='rights_holder', columns='classes_wild', values='seq_id')

s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu")

s.set_title('Number of wild animals observed by each rights holder (empty class removed)', size=16)

plt.show()
try:

    train_df['date_time'] = pd.to_datetime(train_df['date_captured'], errors='coerce')

    train_df["year"] = train_df['date_time'].dt.year

    train_df["month"] = train_df['date_time'].dt.month

    train_df["day"] = train_df['date_time'].dt.day

    train_df["hour"] = train_df['date_time'].dt.hour

    train_df["minute"] = train_df['date_time'].dt.minute

except Exception as ex:

    print("Exception:".format(ex))   
train_df.head()
plot_classes('year', fs=3)
plot_classes('month', fs=5)
plot_classes('day', fs=10)
plot_classes('hour', fs=8)
fig, ax = plt.subplots(1,1,figsize=(16,10))

t = pd.DataFrame(train_df.groupby(['classes_wild', 'hour'])['seq_id'].count().reset_index())

m = t.pivot(index='hour', columns='classes_wild', values='seq_id')

s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu")

s.set_title('Number of wild animals observed per hour', size=16)

plt.show()
tmp = train_df[train_df['classes_wild'] != 'empty']

fig, ax = plt.subplots(1,1,figsize=(16,12))

t = pd.DataFrame(tmp.groupby(['classes_wild', 'hour'])['seq_id'].count().reset_index())

m = t.pivot(index='hour', columns='classes_wild', values='seq_id')

s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu")

s.set_title('Number of wild animals observed per hour', size=16)

plt.show()
fig, ax = plt.subplots(1,1,figsize=(16,8))

t = pd.DataFrame(tmp.groupby(['classes_wild', 'month'])['seq_id'].count().reset_index())

m = t.pivot(index='month', columns='classes_wild', values='seq_id')

s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=True, cmap="YlGnBu")

s.set_title('Number of wild animals observed per month', size=16)

plt.show()
classes = train_df.classes_wild.unique()

fig, ax = plt.subplots(7,2,figsize=(20,28))

i = 0

for class_wild in classes:

    i = i + 1

    plt.subplot(7,2,i)

    tmp = train_df[train_df['classes_wild'] == class_wild]

    t = pd.DataFrame(tmp.groupby(['month', 'hour'])['seq_id'].count().reset_index())

    m = t.pivot(index='hour', columns='month', values='seq_id')

    s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=False, cmap="Greens")

    if(i<13):

        s.set_xlabel('')    

    s.set_title(class_wild, size=12)



plt.show()
classes = train_df.classes_wild.unique()

fig, ax = plt.subplots(7,2,figsize=(16,24))

i = 0

for class_wild in classes:

    i = i + 1

    plt.subplot(7,2,i)

    tmp = train_df[train_df['classes_wild'] == class_wild]

    t = pd.DataFrame(tmp.groupby(['rights_holder', 'month'])['seq_id'].count().reset_index())

    m = t.pivot(index='rights_holder', columns='month', values='seq_id')

    s = sns.heatmap(m, linewidths=.1, linecolor='black', annot=False, cmap="Blues")

    if(i<13):

        s.set_xlabel('')    

    s.set_title(class_wild, size=12)



plt.show()
def draw_category_images(var,cols=5):

    categories = (train_df.groupby([var])[var].nunique()).index

    f, ax = plt.subplots(nrows=len(categories),ncols=cols, figsize=(3*cols,3*len(categories)))

    # draw a number of images for each location

    for i, cat in enumerate(categories):

        sample = train_df[train_df[var]==cat].sample(cols)

        for j in range(0,cols):

            file=IMAGE_PATH + sample.iloc[j]['file_name']

            im = Image.open(file)

            ax[i, j].imshow(im, resample=True)

            ax[i, j].set_title(cat, fontsize=9)  

    plt.tight_layout()

    plt.show()
IMAGE_PATH = os.path.join(PATH,'train_images/')

draw_category_images('classes_wild')
IMAGE_PATH = os.path.join(PATH,'test_images/')

f, ax = plt.subplots(nrows=5,ncols=5, figsize=(15,15))



for i in range(5):

    sample = test_df.sample(5)

    for j in range(5):

        file=IMAGE_PATH + sample.iloc[j]['file_name']

        im = Image.open(file)

        ax[i, j].imshow(im, resample=True)

        ax[i, j].set_title('Not labeled', fontsize=9)  

plt.tight_layout()

plt.show()
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255.

x_test /= 255.
class Metrics(Callback):

    def on_train_begin(self, logs={}):

        self.val_f1s = []

        self.val_recalls = []

        self.val_precisions = []



    def on_epoch_end(self, epoch, logs={}):

        X_val, y_val = self.validation_data[:2]

        y_pred = self.model.predict(X_val)



        y_pred_cat = keras.utils.to_categorical(

            y_pred.argmax(axis=1),

            num_classes=14

        )



        _val_f1 = f1_score(y_val, y_pred_cat, average='macro')

        _val_recall = recall_score(y_val, y_pred_cat, average='macro')

        _val_precision = precision_score(y_val, y_pred_cat, average='macro')



        self.val_f1s.append(_val_f1)

        self.val_recalls.append(_val_recall)

        self.val_precisions.append(_val_precision)



        print((f"val_f1: {_val_f1:.4f}"

               f" — val_precision: {_val_precision:.4f}"

               f" — val_recall: {_val_recall:.4f}"))



        return



f1_metrics = Metrics()
model_densenet = DenseNet121(

    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    input_shape=(32,32,3)

)
model = Sequential()

model.add(model_densenet)

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(cnt_classes_images, activation='softmax'))
model.compile(

    loss='categorical_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)
checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_acc', 

    verbose=1, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)
model.summary()
BATCH_SIZE = 64

EPOCHS = 35

VALID_SPLIT = 0.1

history = model.fit(

    x=x_train,

    y=y_train,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks=[checkpoint, f1_metrics],

    validation_split=VALID_SPLIT

)
with open('history.json', 'w') as f:

    json.dump(history.history, f)

h_df = pd.DataFrame(history.history)

h_df['val_f1'] = f1_metrics.val_f1s

h_df['val_precision'] = f1_metrics.val_precisions

h_df['val_recall'] = f1_metrics.val_recalls

epochs = range(len(h_df['val_f1']))

plt.figure()

fig, ax = plt.subplots(1,3,figsize=(18,4))

ax[0].plot(epochs,h_df['loss'], label='Training loss')

ax[0].plot(epochs,h_df['val_loss'], label='Validation loss')

ax[0].set_title('Training and validation loss')

ax[0].legend()

ax[1].plot(epochs,h_df['acc'],label='Training accuracy')

ax[1].plot(epochs,h_df['val_acc'], label='Validation accuracy')

ax[1].set_title('Training and validation accuracy')

ax[1].legend()

ax[2].plot(epochs,h_df['val_f1'],label='Validation f1-score')

ax[2].plot(epochs,h_df['val_precision'],label='Validation precision')

ax[2].plot(epochs,h_df['val_recall'],label='Validation recall')

ax[2].set_title('Validation f1-score, precision & recall')

ax[2].legend()

plt.show()
model.load_weights('model.h5')

#prepare prediction

y_test = model.predict(x_test)
#submission

submission_df = pd.read_csv(os.path.join(PATH,'sample_submission.csv'))

submission_df['Predicted'] = y_test.argmax(axis=1)



print(submission_df.shape)

submission_df.head(3)
submission_df.to_csv("submission.csv", index=False)