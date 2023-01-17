import numpy as np

import pandas as pd

import os

import glob

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

IMG_SIZE = 224

BATCH_SIZE = 32

NO_EPOCHS = 50

NUM_CLASSES = 2

DATA_FOLDER = "/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images"

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from tqdm import tqdm

import cv2 as cv

from random import shuffle 

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from tensorflow.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

import tensorflow_addons as tfa

import tensorflow as tf

%matplotlib inline 
data_df = pd.read_excel(open("/kaggle/input/ocular-disease-recognition-odir5k/ODIR-5K/data.xlsx", 'rb'), sheet_name='Sheet1')  
data_df.columns = ["id", 'age', "sex", "left_fundus", "right_fundus", "left_diagnosys", "right_diagnosys", "normal",

                  "diabetes", "glaucoma", "cataract", "amd", "hypertension", "myopia", "other"]
print(data_df.loc[(data_df.cataract==1)].shape)

print(data_df.loc[data_df.cataract==0].shape)
data_df.loc[(data_df.cataract==1)]['left_diagnosys'].value_counts()
data_df.loc[(data_df.cataract==1)]['right_diagnosys'].value_counts()
def has_cataract_mentioned(text):

    if 'cataract' in text:

        return 1

    else:

        return 0
data_df['le_cataract'] = data_df['left_diagnosys'].apply(lambda x: has_cataract_mentioned(x))

data_df['re_cataract'] = data_df['right_diagnosys'].apply(lambda x: has_cataract_mentioned(x))
cataract_le_list = data_df.loc[(data_df.cataract==1) & (data_df.le_cataract==1)]['left_fundus'].values

cataract_re_list = data_df.loc[(data_df.cataract==1) & (data_df.re_cataract==1)]['right_fundus'].values

print(len(cataract_le_list), len(cataract_re_list))

non_cataract_le_list = data_df.loc[(data_df.cataract==0) & (data_df.left_diagnosys=="normal fundus")]['left_fundus'].sample(150, random_state=314).values

non_cataract_re_list = data_df.loc[(data_df.cataract==0) & (data_df.right_diagnosys=="normal fundus")]['right_fundus'].sample(150, random_state=314).values

print(len(non_cataract_le_list), len(non_cataract_re_list))
cataract_list = np.concatenate((cataract_le_list, cataract_re_list), axis = 0)

non_cataract_list = np.concatenate((non_cataract_le_list, non_cataract_re_list), axis = 0)

print(len(non_cataract_list), len(cataract_list))
print(len(os.listdir(DATA_FOLDER)))
def label_image(label):

    if label == 1:

        return [1,0]

    elif label == 0: 

        return [0,1]



def process_data(data_image_list, DATA_FOLDER, is_cataract):

    data_df = []

    for img in tqdm(data_image_list):

        path = os.path.join(DATA_FOLDER,img)

        label = label_image(is_cataract)

        img = cv.imread(path,cv.IMREAD_COLOR)

        img = cv.resize(img, (IMG_SIZE,IMG_SIZE))

        data_df.append([np.array(img),np.array(label)])

    shuffle(data_df)

    return data_df
cat_df = process_data(cataract_list, DATA_FOLDER, 1)
cat_no_df = process_data(non_cataract_list, DATA_FOLDER, 0)
def show_images(data, isTest=False):

    f, ax = plt.subplots(5,5, figsize=(15,15))

    for i,data in enumerate(data[:25]):

        img_num = data[1]

        img_data = data[0]

        label = np.argmax(img_num)

        if label  == 0: 

            str_label='Cataract'

        elif label == 1: 

            str_label='No Cataract'

        if(isTest):

            str_label="None"

        ax[i//5, i%5].imshow(img_data)

        ax[i//5, i%5].axis('off')

        ax[i//5, i%5].set_title("Label: {}".format(str_label))

    plt.show()



show_images(cat_df)
show_images(cat_no_df)
train = cat_df + cat_no_df

shuffle(train)

show_images(train)
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)

y = np.array([i[1] for i in train])
model = Sequential()

model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))

model.add(Dense(NUM_CLASSES, activation='softmax'))

# ResNet-50 model is already trained, should not be trained

model.layers[0].trainable = True
opt = tfa.optimizers.LazyAdam()

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.025)

model.compile(optimizer='sgd', loss=loss, metrics=['accuracy'])
model.summary()
plot_model(model, to_file='model.png')

SVG(model_to_dot(model).create(prog='dot', format='svg'))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
train_model = model.fit(X_train, y_train,

                  batch_size=BATCH_SIZE,

                  epochs=NO_EPOCHS,

                  verbose=1,

                  validation_data=(X_val, y_val))
def plot_accuracy_and_loss(train_model):

    hist = train_model.history

    acc = hist['accuracy']

    val_acc = hist['val_accuracy']

    loss = hist['loss']

    val_loss = hist['val_loss']

    epochs = range(len(acc))

    f, ax = plt.subplots(1,2, figsize=(14,6))

    ax[0].plot(epochs, acc, 'g', label='Training accuracy')

    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')

    ax[0].set_title('Training and validation accuracy')

    ax[0].legend()

    ax[1].plot(epochs, loss, 'g', label='Training loss')

    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')

    ax[1].set_title('Training and validation loss')

    ax[1].legend()

    plt.show()

plot_accuracy_and_loss(train_model)
score = model.evaluate(X_val, y_val, verbose=0)

print('Validation loss:', score[0])

print('Validation accuracy:', score[1])
#get the predictions for the test data

predicted_classes = model.predict_classes(X_val)

#get the indices to be plotted

y_true = np.argmax(y_val,axis=1)
correct = np.nonzero(predicted_classes==y_true)[0]

incorrect = np.nonzero(predicted_classes!=y_true)[0]
target_names = ["Cataract", "Normal"]

print(classification_report(y_true, predicted_classes, target_names=target_names))