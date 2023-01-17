import os

import cv2 # computer vision

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

matplotlib.rcParams['image.cmap'] = 'bone'

import pydicom

from glob import glob

from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score

from sklearn.model_selection import train_test_split

from skimage.exposure import equalize_adapthist
seed=42

version=37

BASE_PATH = '../input/rsna-intracranial-hemorrhage-detection/'

TRAIN_DIR = 'rsna-intracranial-hemorrhage-detection/stage_2_train/'

TEST_DIR = 'rsna-intracranial-hemorrhage-detection/stage_2_test/'

DICOM_PATH = '../input/dicom-dataframe/df_trn.fth' # Obtained from https://www.kaggle.com/jhoward/creating-a-metadata-dataframe-fastai/output
df = pd.read_csv(BASE_PATH + '/rsna-intracranial-hemorrhage-detection/stage_2_train.csv').rename(columns={'Label': 'label'})

df[['id', 'img', 'subtype']] = df['ID'].str.split('_', n=3, expand=True)

df['filename'] = df['id'] + '_' + df['img'] + '.dcm'

df['path'] = BASE_PATH + TRAIN_DIR + df['filename']



bdf = pd.read_feather(DICOM_PATH).rename(columns={'SOPInstanceUID': 'filename'}).assign(filename=lambda x: x['filename']+'.dcm')[['filename', 'img_pct_window']]

df = df.merge(bdf, on='filename', how='left').drop(['ID'], axis=1)

df.head()
df['label'].value_counts().plot(kind='bar', grid=True)
df[df['label'] == 1]['subtype'].value_counts().plot(kind='bar', grid=True, colors='blue')
brain_threshold = 0.15

df = df[(df['img_pct_window'] > brain_threshold)].reset_index(drop=True)

print(df.shape)

df.head()
def window_correction(dcm, window_center, window_width):

    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):

        dcm = fix_error_img(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    img = np.clip(img, img_min, img_max)

    return img



def fix_error_img(dcm):

    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: 

        return dcm

    x = dcm.pixel_array + 1000

    px_mode = 4096

    x[x>=px_mode] = x[x>=px_mode] - px_mode

    dcm.PixelData = x.tobytes()

    dcm.RescaleIntercept = -1000

    return dcm



def image_preprocessing(img, target_size=(256, 256)):

    brain_img = window_correction(img, 40, 80)

    subdural_img = window_correction(img, 80, 200)

    soft_img = window_correction(img, 40, 380)

    

    brain_img = (brain_img - 0) / 80

    subdural_img = (subdural_img - (-20)) / 200

    soft_img = (soft_img - (-150)) / 380

    

    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

    bsb_img = cv2.resize(bsb_img, target_size[:2], interpolation=cv2.INTER_LINEAR)

    return bsb_img
one_df = df[df['label'] == 1]

row=1; col=3;

fig = plt.figure(figsize=(10, 10)) #ID_39e5d2a39.dcm ID_f14b31779.dcm ID_330d2fd8c ID_ff776bf6b.dcm ID_8e373a29e

smp1 = []

for i in range(3):

    sample=np.random.randint(one_df.shape[0])

    smp1.append(sample)

    fig.add_subplot(row, col, i+1)

    data = one_df.iloc[sample]

    img = image_preprocessing(pydicom.dcmread(data['path']))

    plt.imshow(img, cmap=plt.cm.bone)

    plt.title(data['filename'] + '_' + data['label'].astype(str))

plt.tight_layout()
zero_df = df[df['label'] == 0]

row=1; col=3;

fig = plt.figure(figsize=(10, 10))

smp0 = []

for i in range(3):

    sample=np.random.randint(zero_df.shape[0])

    smp0.append(sample)

    fig.add_subplot(row, col, i+1)

    data = zero_df.iloc[sample]

    img = image_preprocessing(pydicom.dcmread(data['path']))

    plt.title(data['filename'] + data['label'].astype(str))

    plt.imshow(img, cmap=plt.cm.bone)
def plot_subtypes(subtype, salt=0, figsize=(10, 5)):

    sample_df = positive_df[positive_df['subtype'] == subtype].sample(5, random_state=seed+salt)['path'].values

    

    fig, ax = plt.subplots(1, 5, figsize=figsize)

    for idx, val in enumerate(sample_df):

        img = image_preprocessing(pydicom.dcmread(val))

        ax[idx].imshow(img)

    plt.show()
positive_df = df[df['label'] == 1]
plot_subtypes('epidural', figsize=(20, 10))
plot_subtypes('intraparenchymal', figsize=(20, 10))
plot_subtypes('intraventricular', figsize=(20, 10))
plot_subtypes('subarachnoid', figsize=(20, 10))
plot_subtypes('any', figsize=(20, 10))
df = df.loc[:, ["label", "subtype", "filename"]].drop_duplicates(['filename', 'subtype'])

df = df.set_index(['filename', 'subtype']).unstack(level=-1).droplevel(0, axis=1)
from sklearn.utils import class_weight



tdf = df[df.sum(axis=1) != 0]

for idx, i in enumerate(df.columns):

    tdf[i] = tdf[i].map({1: idx})

    

tdf = tdf.fillna(0)

tdf.head()



ytdf = tdf.values.flatten('f')

del tdf



cw = class_weight.compute_class_weight('balanced', classes=np.unique(ytdf), y=ytdf)

print(cw)
print(df.sum() / df.shape[0])

healthy_images = (df.sum(axis=1) == 0)

print(f'\nHealthy images: {healthy_images.sum()}')
df.head()
test_df = pd.read_csv(BASE_PATH + '/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv')

test_df["Image"] = test_df["ID"].str.slice(stop=12)

test_df["Diagnosis"] = test_df["ID"].str.slice(start=13)



test_df = test_df.loc[:, ["Label", "Diagnosis", "Image"]]

test_df = test_df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
import keras.applications as ka

from keras.models import Sequential, Model, load_model, Input

from keras.optimizers import Adam

from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout, GlobalAveragePooling2D

from keras.models import Model, load_model

from keras.preprocessing import image

from keras.utils import layer_utils, Sequence

from keras.wrappers.scikit_learn import KerasClassifier

from keras.preprocessing.image import ImageDataGenerator

from keras.initializers import glorot_uniform

import keras.backend as K

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau, Callback

from imgaug import augmenters as iaa
DENSE = 128

DROPOUT = 0.4

LEARNING_RATE = 2e-3

BATCH_SIZE = 32

EPOCHS = 6

N_CLASS = 6



target_size=(224, 224, 3)
def weighted_categorical_loss(y_true, y_pred):

    class_weights = np.array([2., 1., 1., 1., 1., 1.])

    eps = K.epsilon()

    y_pred = K.clip(y_pred, eps, 1.0-eps)



    out = -(y_true  * K.log(y_pred) * class_weights + (1.0 - y_true) * K.log(1.0 - y_pred) * class_weights)

    return K.mean(out, axis=-1)
class DataGenerator(Sequence):



    def __init__(self, list_IDs, labels=None, batch_size=1, img_size=target_size, img_dir='rsna-intracranial-hemorrhage-detection/stage_2_train/'):



        self.list_IDs = list_IDs

        self.labels = labels

        self.batch_size = batch_size

        self.img_size = target_size

        self.img_dir= BASE_PATH + img_dir

        self.on_epoch_end()



    def __len__(self):

        return int(np.ceil(len(self.indices) / self.batch_size))



    def __getitem__(self, index):

        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indices]

        

        if self.labels is not None:

            X, Y = self.__data_generation(list_IDs_temp)

            return X, Y

        else:

            X = self.__data_generation(list_IDs_temp)

            return X

        

    def on_epoch_end(self):

        if self.labels is not None: # for training phase we undersample and shuffle

            # keep probability of any=0 and any=1

            keep_prob = self.labels.iloc[:, 0].map({0: 0.35, 1: 0.5})

            keep = (keep_prob > np.random.rand(len(keep_prob)))

            self.indices = np.arange(len(self.list_IDs))[keep]

            np.random.shuffle(self.indices)

        else:

            self.indices = np.arange(len(self.list_IDs))



    def read_dicom(self, path, target_size):

        dcm = pydicom.dcmread(path)



        try:

            img = image_preprocessing(dcm, target_size=target_size)

        except:

            img = np.zeros(target_size)



        return img

            

    def augment_img(self, image): 

        augment_img = iaa.Sequential([

            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 0.5))),

            iaa.Fliplr(0.25),

            iaa.Flipud(0.25)])

        image_aug = augment_img.augment_image(image)

        return image_aug



    def __data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.img_size))

        

        if self.labels is not None: # training phase

            Y = np.empty((self.batch_size, 6), dtype=np.float32)

        

            for i, ID in enumerate(list_IDs_temp):

                #X[i,] = self.augment_img(self.read_dicom(self.img_dir+ID, self.img_size))

                X[i,] = self.read_dicom(self.img_dir+ID, self.img_size)

                Y[i,] = self.labels.loc[ID].values

        

            return X, Y

        

        else:

            for i, ID in enumerate(list_IDs_temp):

                X[i,] = self.read_dicom(self.img_dir+ID+'.dcm', self.img_size)

            

            return X
class PredictionCheckpoint(Callback):

    

    def __init__(self, test_df, batch_size=BATCH_SIZE, input_size=target_size):

        

        self.test_df = test_df

        self.test_images_dir = BASE_PATH + TEST_DIR

        self.batch_size = batch_size

        self.input_size = input_size

        

    def on_train_begin(self, logs={}):

        self.test_predictions = []

        

    def on_epoch_end(self,batch, logs={}):

        self.test_predictions.append(

            self.model.predict_generator(

                DataGenerator(self.test_df.index, None, self.batch_size, self.input_size, img_dir=TEST_DIR), verbose=2)[:len(self.test_df)])
def build_model(input_shape, pretrained_model=None):

    net = pretrained_model(include_top=False, input_shape=input_shape, weights='imagenet')

    

    model = Sequential()

    model.add(net)

    model.add(GlobalAveragePooling2D())

    #model.add(BatchNormalization())

    #model.add(Dropout(DROPOUT))

    

    #model.add(Dense(DENSE, activation='relu'))

    #model.add(BatchNormalization())

    #model.add(Dropout(DROPOUT-0.3))

    

    model.add(Dense(N_CLASS, activation='sigmoid'))

    model.summary()

    

    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[weighted_categorical_loss])

    return model
train_idx, valid_idx = train_test_split(df.index, test_size=0.01, random_state=42)
def lr_scheduler(epoch, lr):

    decay_rate = 0.85

    decay_step = 1

    if epoch % decay_step == 0 and epoch:

        return lr * pow(decay_rate, np.floor(epoch / decay_step))

    return lr
train_generator = DataGenerator(train_idx, labels=df.loc[train_idx], batch_size=BATCH_SIZE, img_size=target_size)

#valid_generator = DataGenerator(valid_idx, labels=df.loc[valid_idx], batch_size=len(valid_idx), img_size=target_size)



model_path = f'InceptionV3_{version}_{EPOCHS}_{DENSE}.h5'

pred_history = PredictionCheckpoint(test_df)



callbacks = [

    LearningRateScheduler(lr_scheduler, verbose=1),

    ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto'),

    pred_history

]

    

model = build_model(pretrained_model=ka.InceptionV3, input_shape=(224, 224, 3))

history = model.fit_generator(train_generator, callbacks=callbacks, 

                              epochs=EPOCHS, verbose=1, use_multiprocessing=True, workers=4, class_weight=cw)
hdf = pd.DataFrame(history.history)

hdf[['loss', 'weighted_categorical_loss']].plot(grid=True, figsize=(15, 3), title='Loss and Accuracy Graphs')
test_df.iloc[:, :] = np.average(pred_history.test_predictions, axis=0, weights=[0, 0, 1, 2, 4, 8])

test_df = test_df.stack().reset_index()

test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])

test_df = test_df.drop(["Image", "Diagnosis"], axis=1)
test_df.to_csv('submission.csv', index=False)