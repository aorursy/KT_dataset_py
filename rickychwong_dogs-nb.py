# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.tree
import sklearn.ensemble

import glob
import random
import base64
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from os.path import join, exists, expanduser
from os import listdir, makedirs

from PIL import Image
from io import BytesIO
from IPython.display import HTML
pd.set_option('display.max_colwidth', -1)

def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((150, 150), Image.LANCZOS)
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'
# Any results you write to the current directory are saved as output.
labels = pd.read_csv('../input/dog-breed-identification/labels.csv')
labels_test=labels.sample(20)
labels_test['file'] = labels_test.id.map(lambda id: f'../input/dog-breed-identification/train/{id}.jpg')
labels_test['image'] = labels_test.file.map(lambda f: get_thumbnail(f))
labels_test.head()
HTML(labels_test[['breed', 'image']].to_html(formatters={'image': image_formatter}, escape=False))
!ls ../input/keras-pretrained-models/
cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    os.makedirs(models_dir)
!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/
!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
!ls ~/.keras/models
nclass = 10
image_size=224

breed_list = list(labels.groupby('breed').count().sort_values(by='id',ascending=False).head(nclass).index)
labels = labels[labels['breed'].isin(breed_list)]
labels['tmp']=1
labels=labels.pivot('id', 'breed', 'tmp').reset_index().fillna(0)
labels.head()
data_dir='../input/dog-breed-identification/'

def read_img(img_id, train_or_test, size):
    """Read and resize image.
    # Arguments
        img_id: string
        train_or_test: string 'train' or 'test'.
        size: resize the original image.
    # Returns
        Image as numpy array.
    """
    img = image.load_img(join(data_dir, train_or_test, '%s.jpg' % img_id), target_size=size)
    img = image.img_to_array(img)
    return img
resnet_weights_path = '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
model.add(Dense(nclass, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
model.layers[0].trainable = False
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
cv_num=10
kfold=sklearn.model_selection.KFold(cv_num,shuffle=True,random_state=42)

x_trains = np.zeros((len(labels), image_size, image_size, 3), dtype='float32')
for i, img_id in tqdm(enumerate(labels['id'])):
    img = read_img(img_id, 'train', (image_size, image_size))
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    x_trains[i] = x

y_trains = labels[breed_list].values    
nepochs = 3
if True:
    for i,(train_index,valid_index) in enumerate(kfold.split(y_trains)):
        x_train = x_trains[train_index]
        y_train = y_trains[train_index]
        x_valid = x_trains[valid_index]
        y_valid = y_trains[valid_index]
        
        datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, validation_data=datagen.flow(x_valid, y_valid, batch_size=32),validation_steps=len(y_valid)/32, epochs=nepochs)
if True:
    x_trains_predict = model.predict(x_trains, verbose=1)
if True:

    failed=[]
    truths=[]
    predictions=[]

    for i in range(y_trains.shape[0]):
        truth=breed_list[np.argmax(y_trains[i])]
        prediction=breed_list[np.argmax(x_trains_predict[i])]
        truths.append(truth)
        predictions.append(prediction)
        if prediction != truth:            
            failed.append(i)
            
    print('Failed: ',len(failed),'/',len(y_trains),'=',len(failed)/len(y_trains))
    
    ndisplay = min(20,len(failed))

    temp=pd.DataFrame()
    temp['Predicted'] = [predictions[i] for i in failed[:ndisplay]]
    temp['Truth'] = [truths[i] for i in failed[:ndisplay]]
    
    temp['file'] = labels[:ndisplay].id.map(lambda id: f'../input/dog-breed-identification/train/{id}.jpg')
    temp['image'] = temp.file.map(lambda f: get_thumbnail(f))




HTML(temp[['Predicted','Truth', 'image']].to_html(formatters={'image': image_formatter}, escape=False))
if True:    
    test_ids = listdir('../input/dog-breed-identification/test')

    test_ids=test_ids[0:20]

    x_tests = np.zeros((len(test_ids), image_size, image_size, 3), dtype='float32')
    for i, img_id in tqdm(enumerate(test_ids)):
    
        img = image.load_img(join(data_dir, 'test', '%s' % img_id), target_size=(image_size, image_size))
        img = image.img_to_array(img)
        x = preprocess_input(np.expand_dims(img.copy(), axis=0))
        x_tests[i] = x


    predicts = model.predict(x_tests)

    preds = []

    for i in range(len(predicts)):
        preds.append(breed_list[np.argmax(predicts[i])])

    sub = pd.DataFrame()
    sub['id'] = test_ids
    sub['breed'] = preds

    sub['file'] = sub.id.map(lambda id: f'../input/dog-breed-identification/test/{id}')
    sub['image'] = sub.file.map(lambda f: get_thumbnail(f))

HTML(sub[['breed', 'image']].to_html(formatters={'image': image_formatter}, escape=False))
