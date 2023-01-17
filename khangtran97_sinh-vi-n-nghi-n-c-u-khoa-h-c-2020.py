# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from tqdm import tqdm

import PIL

from PIL import Image, ImageOps

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy, categorical_crossentropy

from keras.applications.resnet50 import preprocess_input

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score, accuracy_score

from keras.utils import Sequence

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix



WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

SIZE = 224

NUM_CLASSES = 2
os.listdir("../input/nckh2020/nckh/nckh/push")
filenames = os.listdir("../input/nckh2020/nckh/nckh")

categories = []

file_path = []

name = []

for filename in filenames:

    category = filename

    for file in os.listdir("../input/nckh2020/nckh/nckh/"+filename):

        file_path.append("../input/nckh2020/nckh/nckh/"+filename+"/"+file)

        name.append(file.split('.')[0])

        if category == 'handshake':

            categories.append(0)

        else:

            categories.append(1)

        

df = pd.DataFrame({

    'id': name,

    'filename': file_path,

    'category': categories

})
df.head()
os.listdir("../input/nckh2020/")
data = pd.read_csv("../input/nckh2020/nckh.csv")

data.head()
data['id'] = data['filename']

data = data.drop(['filename'], axis = 1)

data.head()
data = data.merge(df, on = 'id', how = 'left')
data.head()
data = data.drop(['Unnamed: 0', 'label'], axis = 1)
data.head()
data = data.sort_values('id')

data.head()
y = data['category']

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify = y)

train_df = train_df.reset_index(drop=True)

test_df = test_df.reset_index(drop=True)
train_csv = train_df.drop(['filename'], axis = 1)

test_csv = test_df.drop(['filename'], axis = 1)
train_csv.head()
train_img = train_df.drop(data.drop(['filename','id', 'category'],axis = 1).columns, axis = 1)

test_img = test_df.drop(data.drop(['filename','id', 'category'],axis = 1).columns, axis = 1)
train_img.head()
train_features = train_csv.drop(['category'],axis=1)

train_targets = train_csv['category']

test_features = test_csv.drop(['category'],axis=1)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
hyper_params = {

    'task': 'train',

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': ['binary_logloss'],

    'learning_rate': 0.01,

    'feature_fraction': 0.85,

    'subsample': 0.8,

    'subsample_freq': 2,

    'verbose': 0,

    "max_depth": 40,

    "num_leaves": 250,  

    "max_bin": 512,

    "num_iterations": 10000,

}
train_targets = pd.DataFrame(train_targets)
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

import math  

from sklearn.model_selection import KFold, StratifiedKFold



# score = []

predict_val = pd.DataFrame(test_csv['id'])

skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=123)

skf.get_n_splits(train_features, train_targets)

oof_lgb_df = pd.DataFrame()

predictions = pd.DataFrame(test_csv['id'])

x_test = test_features.drop(['id'], axis = 1)



for fold, (trn_idx, val_idx) in enumerate(skf.split(train_features, train_targets)):

    x_train, y_train = train_features.iloc[trn_idx], train_targets.iloc[trn_idx]['category']

    x_valid, y_valid = train_features.iloc[val_idx], train_targets.iloc[val_idx]['category']

    index = x_valid['id']

    x_train = x_train.drop(['id'], axis = 1)

    x_valid = x_valid.drop(['id'], axis = 1)

    p_valid = 0

    yp = 0

    yv = 0

    gbm = lgb.LGBMRegressor(**hyper_params)

    gbm.fit(x_train, y_train,

        eval_set=[(x_valid, y_valid)],

        eval_metric='binary_logloss',

        verbose = 500,

        early_stopping_rounds=100)

#     score.append(math.sqrt(mean_squared_error(gbm.predict(x_valid), y_valid)))

    yp += gbm.predict(x_test)

    yv += gbm.predict(x_valid)

    fold_pred = pd.DataFrame({'id': index,

                              'label':gbm.predict(x_valid)})

    oof_lgb_df = pd.concat([oof_lgb_df, fold_pred], axis=0)

    

    predictions['fold{}'.format(fold+1)] = yp
oof_lgb_df = oof_lgb_df.sort_values('id')

oof_lgb_df.head()
lgb_predict = pd.DataFrame()

lgb_predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5'])/5
lgb = lgb_predict['predict'].round(0)
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

print(accuracy_score(test_df['category'], lgb))

print(f1_score(test_df['category'], lgb, average='macro'))
train_features = train_img.drop(['category'],axis=1)

train_targets = train_img['category']

test_features = test_img.drop(['category'],axis=1)



train_targets = pd.DataFrame(train_targets)

# train_targets = to_categorical(train_targets, num_classes=NUM_CLASSES)
# https://github.com/aleju/imgaug

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([

    sometimes(

        iaa.OneOf([

            iaa.Add((-10, 10), per_channel=0.5),

            iaa.Multiply((0.9, 1.1), per_channel=0.5),

            iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5)

        ])

    ),

    iaa.Fliplr(0.5),

    # iaa.Crop(percent=(0, 0.1)),

    # iaa.Flipud(0.5)

],random_order=True)
class My_Generator(Sequence):



    def __init__(self, image_filenames, labels,

                 batch_size, is_train=False,

                 mix=False, augment=False):

        self.image_filenames, self.labels = image_filenames, labels

        self.batch_size = batch_size

        self.is_train = is_train

        self.is_augment = augment

        if(self.is_train):

            self.on_epoch_end()

        self.is_mix = mix



    def __len__(self):

        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]



        if(self.is_train):

            return self.train_generate(batch_x, batch_y)

        return self.valid_generate(batch_x, batch_y)



    def on_epoch_end(self):

        if(self.is_train):

            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

    

    def mix_up(self, x, y):

        lam = np.random.beta(0.2, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)        

        

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        

        return mixed_x, mixed_y



    def train_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x['filename'], batch_y):

            img = cv2.imread(sample)

#             print('../input/data/Data'+sample)

            img = cv2.resize(img, (SIZE, SIZE))

#             print(img.shape)

            if(self.is_augment):

                img = seq.augment_image(img)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        # batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y



    def valid_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x['filename'], batch_y):

            img = cv2.imread(sample)

#             print(img)

            img = cv2.resize(img, (SIZE, SIZE))

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        # batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)

from keras.applications.resnet50 import ResNet50

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras

from keras.models import Model
function = "softmax"

def create_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    base_model = ResNet50(include_top=False,

                   weights=None,

                   input_tensor=input_tensor)

    base_model.load_weights('../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    x = GlobalAveragePooling2D()(base_model.output)

#     x = Dropout(0.5)(x)

#     x = Dense(1024, activation='relu')(x)

    x = Dropout(0.3)(x)

    final_output = Dense(n_out, activation=function, name='final_output')(x)

    model = Model(input_tensor, final_output)

    

    return model
from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)



epochs = 80; batch_size = 32

checkpoint = ModelCheckpoint('../working/Resnet50-visible.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 

                                   verbose=1, mode='min', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=9)



# csv_logger = CSVLogger(filename='../working/training_log.csv',

#                        separator=',',

#                        append=True)

callbacks_list = [checkpoint, reduceLROnPlat, early]



# train_generator = My_Generator(train_x, train_y, 128, is_train=True)

# train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=False, augment=True)

# valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)



# model = create_model(

#     input_shape=(SIZE,SIZE,3), 

#     n_out=NUM_CLASSES)
from sklearn.metrics import mean_squared_error

import math  

from sklearn.model_selection import KFold, StratifiedKFold



# score = []

predict_val = pd.DataFrame(test_img['id'])

skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=123)

skf.get_n_splits(train_features, train_targets)

oof_nn_df = pd.DataFrame()

predictions = pd.DataFrame(test_img['id'])

x_test = test_features

# cat_features = ['artist_id','singer_0','composers_0','composers_id','hour','dayofweek','quarter','month','year','dayofyear', 'dayofmonth', 'weekofyear']





for fold, (trn_idx, val_idx) in enumerate(skf.split(train_features, train_targets)):

    x_train, y_train = train_features.iloc[trn_idx], train_targets.iloc[trn_idx]['category']

    x_valid, y_valid = train_features.iloc[val_idx], train_targets.iloc[val_idx]['category']

    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)

    y_valid = to_categorical(y_valid, num_classes=NUM_CLASSES)

    index = x_valid['id']

    x_train = x_train.drop(['id'], axis = 1)

    x_valid = x_valid.drop(['id'], axis = 1)

    p_valid = 0

    yp = 0

    yv = 0

    

    train_generator = My_Generator(x_train, y_train, 128, is_train=True)

    train_mixup = My_Generator(x_train, y_train, batch_size, is_train=True, mix=False, augment=True)

    valid_generator = My_Generator(x_valid, y_valid, batch_size, is_train=False)



    model = create_model(

        input_shape=(SIZE,SIZE,3), 

        n_out=NUM_CLASSES)

    

    for layer in model.layers:

        layer.trainable = False



    for i in range(-3,0):

        model.layers[i].trainable = True



    model.compile(

        loss='categorical_crossentropy',

#         loss='binary_crossentropy',

        optimizer=Adam(1e-3))



    model.fit_generator(train_generator,steps_per_epoch=np.ceil(float(len(y_train)) / float(128)),epochs=2,workers=WORKERS, use_multiprocessing=True,verbose=1)

    

    

    # train all layers

    for layer in model.layers:

        layer.trainable = True



    callbacks_list = [checkpoint, reduceLROnPlat, early]

    model.compile(loss='categorical_crossentropy',

                # loss=kappa_loss,

                # loss='binary_crossentropy',

                optimizer=Adam(lr=1e-4),

    #             optimizer=AdamAccumulate(lr=1e-4, accum_iters=2),

                metrics=['accuracy'])



    model.fit_generator(

        train_mixup,

        steps_per_epoch=np.ceil(float(len(x_train)) / float(batch_size)),

        validation_data=valid_generator,

        validation_steps=np.ceil(float(len(x_valid)) / float(batch_size)),

        epochs=epochs,

        verbose=1,

        workers=1, use_multiprocessing=False,

        callbacks=callbacks_list)



    

#     score.append(math.sqrt(mean_squared_error(model.predict(x_valid), y_valid)))

    predicted = []

    for sample in x_valid['filename']:

    #     path = os.path.join("../input/final-thermal/data/"+sample)

        image = cv2.imread(sample)

        image = cv2.resize(image, (SIZE, SIZE))

        score_predict = model.predict((image[np.newaxis])/255)

        score_predict = np.argmax(score_predict)

        label_predict = score_predict.astype(int).sum() - 1

        predicted.append(float(label_predict))

        

    fold_pred = pd.DataFrame({'ID': index,

                              'label': predicted})

    

    

    predicted = []

    for sample in x_test['filename']:

    #     path = os.path.join("../input/final-thermal/data/"+sample)

        image = cv2.imread(sample)

        image = cv2.resize(image, (SIZE, SIZE))

        score_predict = model.predict((image[np.newaxis])/255)

        score_predict = np.argmax(score_predict)

        label_predict = score_predict.astype(int).sum() - 1

        predicted.append(float(label_predict))    



    oof_nn_df = pd.concat([oof_nn_df, fold_pred], axis=0)

    predictions['fold{}'.format(fold+1)] = predicted
oof_nn_df
nn_predict = pd.DataFrame()

nn_predict['predict'] = (predictions['fold1']+predictions['fold2']+predictions['fold3']+predictions['fold4']+predictions['fold5'])/5

oof_nn_df = oof_nn_df.sort_values('ID')
nn_predict.head(20)
nn = np.absolute(nn_predict['predict'].round(0))
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

print(accuracy_score(test_df['category'], nn))

print(f1_score(test_df['category'], nn, average='macro'))
oof_lgb_df
oof_data = None

oof_data = pd.DataFrame({ 'lgbm': oof_lgb_df['label'],

                          'nn': oof_nn_df['label'],

                          'label': train_targets['category']

})

oof_data.head()
oof_test = pd.DataFrame({ 'lgbm': lgb_predict['predict'],

#                           'svm': svm_predict['predict'],

                          'nn': nn_predict['predict']

})

oof_test.head()
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(oof_data.drop(['label'], axis = 1), oof_data['label'])
prediction = clf.predict(oof_test)

prediction = pd.DataFrame({

    'label': prediction})

prediction.head()
test_df
from sklearn.metrics import accuracy_score

accuracy_score(test_df['category'], prediction['label'])
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

print(accuracy_score(test_df['category'], prediction['label']))

print(f1_score(test_df['category'], prediction['label'], average='macro'))
import seaborn as sn

print('Confusion Matrix')

cm = confusion_matrix(test_df['category'], prediction['label'])

print(cm)

sn.set(font_scale=1.4)#for label size

sn.heatmap(cm, annot=True,annot_kws={"size": 16})# font size