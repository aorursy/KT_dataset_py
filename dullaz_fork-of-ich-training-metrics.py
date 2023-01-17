import os

import json



import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import pydicom



from keras import layers

from keras.applications import DenseNet121, ResNet50V2, InceptionV3

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

from keras.initializers import Constant

from keras.utils import Sequence

from keras.models import Sequential

from keras.optimizers import Adam

from keras.models import Model, load_model

from keras.layers import GlobalAveragePooling2D, Dense, Activation, concatenate, Dropout

from keras.initializers import glorot_normal, he_normal

from keras.regularizers import l2



import keras.metrics as M

import tensorflow_addons as tfa

import pickle



from keras import backend as K



import tensorflow as tf

from tensorflow.python.ops import array_ops



from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold



import warnings

warnings.filterwarnings(action='once')
BASE_PATH = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'

TRAIN_DIR = 'stage_2_train/'

TEST_DIR = 'stage_2_test/'
train_df = pd.read_csv(BASE_PATH + 'stage_2_train.csv')

#sub_df = pd.read_csv(BASE_PATH + 'stage_1_sample_submission.csv')



train_df['id'] = train_df['ID'].apply(lambda st: "ID_" + st.split('_')[1])

train_df['subtype'] = train_df['ID'].apply(lambda st: st.split('_')[2])

#sub_df['filename'] = sub_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")

#sub_df['type'] = sub_df['ID'].apply(lambda st: st.split('_')[2])



print(train_df.shape)

train_df.head()
train_df = train_df[["id","subtype","Label"]]

train_df.head()
train_df = pd.pivot_table(train_df,index="id",columns="subtype",values="Label")

train_df.head()
pivot_df = train_df.copy()

#bad = []

#for index,row in tqdm(pivot_df.iterrows()):

#    f = BASE_PATH+TRAIN_DIR+index+".dcm"

#    dcm = pydicom.dcmread(f)

#    try:

#        d = dcm.pixel_array

#    except:

#        bad.append(index)

pivot_df.drop("ID_6431af929",inplace=True)

#print(bad)
def map_to_gradient(grey_img):

    rainbow_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))

    rainbow_img[:, :, 0] = np.clip(4 * grey_img - 2, 0, 1.0) * (grey_img > 0) * (grey_img <= 1.0)

    rainbow_img[:, :, 1] =  np.clip(4 * grey_img * (grey_img <=0.75), 0,1) + np.clip((-4*grey_img + 4) * (grey_img > 0.75), 0, 1)

    rainbow_img[:, :, 2] = np.clip(-4 * grey_img + 2, 0, 1.0) * (grey_img > 0) * (grey_img <= 1.0)

    return rainbow_img



def rainbow_window(dcm):

    grey_img = window_image(dcm, 40, 80)

    return map_to_gradient(grey_img)



import cupy as cp



def sigmoid_window(dcm, window_center, window_width, U=1.0, eps=(1.0 / 255.0)):

    img = dcm.pixel_array

    img = cp.array(np.array(img))

    _, _, intercept, slope = get_windowing(dcm)

    img = img * slope + intercept

    ue = cp.log((U / eps) - 1.0)

    W = (2 / window_width) * ue

    b = ((-2 * window_center) / window_width) * ue

    z = W * img + b

    img = U / (1 + cp.power(np.e, -1.0 * z))

    img = (img - cp.min(img)) / (cp.max(img) - cp.min(img))

    return cp.asnumpy(img)



def sigmoid_bsb_window(dcm):

    brain_img = sigmoid_window(dcm, 40, 80)

    subdural_img = sigmoid_window(dcm, 80, 200)

    bone_img = sigmoid_window(dcm, 600, 2000)

    

    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))

    bsb_img[:, :, 0] = brain_img

    bsb_img[:, :, 1] = subdural_img

    bsb_img[:, :, 2] = bone_img

    return bsb_img



def window_image(dcm, window_center, window_width):

    _, _, intercept, slope = get_windowing(dcm)

    img = dcm.pixel_array * slope + intercept

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    img[img < img_min] = img_min

    img[img > img_max] = img_max

    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return img



def bsb_window(dcm):

    brain_img = window_image(dcm, 40, 80)

    subdural_img = window_image(dcm, 80, 200)

    bone_img = window_image(dcm, 600, 2000)

    

    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))

    bsb_img[:, :, 0] = brain_img

    bsb_img[:, :, 1] = subdural_img

    bsb_img[:, :, 2] = bone_img

    return bsb_img

    

def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
def preprocess(file,type="WINDOW"):

    dcm = pydicom.dcmread(BASE_PATH+TRAIN_DIR+file+".dcm")

    if type == "WINDOW":

        window_center , window_width, intercept, slope = get_windowing(dcm)

        w = window_image(dcm, window_center, window_width)

        win_img = np.repeat(w[:, :, np.newaxis], 3, axis=2)

        #return win_img

    elif type == "SIGMOID":

        window_center , window_width, intercept, slope = get_windowing(dcm)

        test_img = dcm.pixel_array

        w = sigmoid_window(dcm, window_center, window_width)

        win_img = np.repeat(w[:, :, np.newaxis], 3, axis=2)

        #return win_img

    elif type == "BSB":

        win_img = bsb_window(dcm)

        #return win_img

    elif type == "SIGMOID_BSB":

        win_img = sigmoid_bsb_window(dcm)

    elif type == "GRADIENT":

        win_img = rainbow_window(dcm)

        #return win_img

    else:

        win_img = dcm.pixel_array

    resized = cv2.resize(win_img,(224,224))

    return resized



class DataLoader(Sequence):

    def __init__(self, dataframe,

                 batch_size,

                 shuffle,

                 input_shape,

                 num_classes=6,

                 steps=None,

                 prep="BSB"):

        

        self.data_ids = dataframe.index.values

        self.dataframe = dataframe

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.input_shape = input_shape

        self.num_classes = num_classes

        self.current_epoch=0

        self.prep = prep

        self.steps=steps

        if self.steps is not None:

            self.steps = np.round(self.steps/3) * 3

            self.undersample()

        

    def undersample(self):

        part = np.int(self.steps/3 * self.batch_size)

        zero_ids = np.random.choice(self.dataframe.loc[self.dataframe["any"] == 0].index.values, size=5000, replace=False)

        hot_ids = np.random.choice(self.dataframe.loc[self.dataframe["any"] == 1].index.values, size=5000, replace=True)

        self.data_ids = list(set(zero_ids).union(hot_ids))

        np.random.shuffle(self.data_ids)

        

    # defines the number of steps per epoch

    def __len__(self):

        if self.steps is None:

            return np.int(np.ceil(len(self.data_ids) / np.float(self.batch_size)))

        else:

            return 3*np.int(self.steps/3) 

    

    # at the end of an epoch: 

    def on_epoch_end(self):

        # if steps is None and shuffle is true:

        if self.steps is None:

            self.data_ids = self.dataframe.index.values

            if self.shuffle:

                np.random.shuffle(self.data_ids)

        else:

            self.undersample()

        self.current_epoch += 1

    

    # should return a batch of images

    def __getitem__(self, item):

        # select the ids of the current batch

        current_ids = self.data_ids[item*self.batch_size:(item+1)*self.batch_size]

        X, y = self.__generate_batch(current_ids)

        return X, y

    

    # collect the preprocessed images and targets of one batch

    def __generate_batch(self, current_ids):

        X = np.empty((self.batch_size, *self.input_shape, 3))

        y = np.empty((self.batch_size, self.num_classes))

        for idx, ident in enumerate(current_ids):

            # Store sample

            #image = self.preprocessor.preprocess(ident) 

            image = preprocess(ident,self.prep)

            X[idx] = image

            # Store class

            y[idx] = self.__get_target(ident)

        return X, y

    

    # extract the targets of one image id:

    def __get_target(self, ident):

        targets = self.dataframe.loc[ident].values

        return targets
def DenseNet():

    densenet = DenseNet121(

    #weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    weights='imagenet',

    include_top=False)

    return densenet

def ResNet():

    resnet = ResNet50V2(weights="imagenet",include_top=False)

    return resnet

def Inception():

    incept = InceptionV3(weights="imagenet",include_top=False)

    return incept



def get_backbone(name):

    if name == "RESNET":

        return ResNet

    elif name == "DENSE":

        return DenseNet

    elif name == "INCEPT":

        return Inception



def build_model(backbone):

    m = backbone()

    x = m.output

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.3)(x)

    x = Dense(100, activation="relu")(x)

    x = Dropout(0.3)(x)

    pred = Dense(6,activation="sigmoid")(x)

    model = Model(inputs=m.input,outputs=pred)

    return model
train,test = train_test_split(pivot_df,test_size=0.2,random_state=42,shuffle=True)



split_seed = 1

kfold = StratifiedKFold(n_splits=5, random_state=split_seed,shuffle=True).split(np.arange(train.shape[0]), train["any"].values)



train_idx, dev_idx = next(kfold)



train_data = train.iloc[train_idx]

dev_data = train.iloc[dev_idx]



#train_data, dev_data = train_test_split(traindf, test_size=0.1, stratify=traindf.values, random_state=split_seed)

print(train_data.shape)

print(dev_data.shape)
def casting_focal_loss():

    def inner_casting(y_true,y_pred):

        y_true = tf.cast(y_true, tf.float32)

        y_pred = tf.cast(y_pred, tf.float32)

        return tfa.losses.SigmoidFocalCrossEntropy(y_true,y_pred)



METRICS = ['categorical_accuracy']

#LOSS = casting_focal_loss()

#LOSS = 'categorical_crossentropy'

LOSSES = [tfa.losses.SigmoidFocalCrossEntropy(),'categorical_crossentropy']

L_NAME = ['cat_cross','focal_loss']

p = 0

BATCH_SIZE = 32

TRAIN_STEPS = 200#(train_data.shape[0] // BATCH_SIZE)

VAL_STEPS = 200#dev_data.shape[0] // BATCH_SIZE

EPOCHS = 15



ALPHA = 0.5

GAMMA = 2



LR = 0.0001



PREP = "SIGMOID"

ARCH = 'RESNET'

LOSS = LOSSES[p]



train_dataloader = DataLoader(train_data,

                              BATCH_SIZE,

                              shuffle=True,

                              input_shape=(224,224),

                              steps=TRAIN_STEPS,

                              prep=PREP)



dev_dataloader = DataLoader(dev_data, 

                            BATCH_SIZE,

                            shuffle=True,

                            input_shape=(224,224),

                            steps=VAL_STEPS,

                            prep=PREP)

test_dataloader = DataLoader(test,

                            BATCH_SIZE,

                            shuffle=False,

                            input_shape=(224,224),

                            prep=PREP)



cpath = "./" + ARCH + "_" + PREP + "_" + str(TRAIN_STEPS) + "_" + str(EPOCHS) + "_" + L_NAME[p]

p += 1

checkpoint = ModelCheckpoint(filepath=cpath + ".model",mode="min",verbose=1,save_best_only=True,save_weights_only=False,period=1)



model = build_model(get_backbone(ARCH))



model.compile(optimizer=Adam(learning_rate=LR),loss=LOSS,metrics=METRICS)



history = model.fit_generator(generator=train_dataloader,validation_data=dev_dataloader,epochs=EPOCHS,workers=8,callbacks=[checkpoint])



with open(cpath + ".history", 'wb') as file_pi:

    pickle.dump(history.history, file_pi)

    

print("Evaluate")

test_prob = model.evaluate(test_dataloader)

res = dict(zip(model.metrics_names, test_prob))

print(res)
def casting_focal_loss():

    def inner_casting(y_true,y_pred):

        y_true = tf.cast(y_true, tf.float32)

        y_pred = tf.cast(y_pred, tf.float32)

        return tfa.losses.SigmoidFocalCrossEntropy(y_true,y_pred)



METRICS = ['categorical_accuracy']

#LOSS = casting_focal_loss()

#LOSS = 'categorical_crossentropy'

LOSSES = [tfa.losses.SigmoidFocalCrossEntropy(),'categorical_crossentropy']

L_NAME = ['cat_cross','focal_loss']

p = 1

BATCH_SIZE = 32

TRAIN_STEPS = 200#(train_data.shape[0] // BATCH_SIZE)

VAL_STEPS = 200#dev_data.shape[0] // BATCH_SIZE

EPOCHS = 15



ALPHA = 0.5

GAMMA = 2



LR = 0.0001



PREP = "SIGMOID"

ARCH = 'RESNET'

LOSS = LOSSES[p]



train_dataloader = DataLoader(train_data,

                              BATCH_SIZE,

                              shuffle=True,

                              input_shape=(224,224),

                              steps=TRAIN_STEPS,

                              prep=PREP)



dev_dataloader = DataLoader(dev_data, 

                            BATCH_SIZE,

                            shuffle=True,

                            input_shape=(224,224),

                            steps=VAL_STEPS,

                            prep=PREP)

test_dataloader = DataLoader(test,

                            BATCH_SIZE,

                            shuffle=False,

                            input_shape=(224,224),

                            prep=PREP)



cpath = "./" + ARCH + "_" + PREP + "_" + str(TRAIN_STEPS) + "_" + str(EPOCHS) + "_" + L_NAME[p]

p += 1

checkpoint = ModelCheckpoint(filepath=cpath + ".model",mode="min",verbose=1,save_best_only=True,save_weights_only=False,period=1)



model = build_model(get_backbone(ARCH))



model.compile(optimizer=Adam(learning_rate=LR),loss=LOSS,metrics=METRICS)



history = model.fit_generator(generator=train_dataloader,validation_data=dev_dataloader,epochs=EPOCHS,workers=8,callbacks=[checkpoint])



with open(cpath + ".history", 'wb') as file_pi:

    pickle.dump(history.history, file_pi)

    

print("Evaluate")

test_prob = model.evaluate(test_dataloader)

res = dict(zip(model.metrics_names, test_prob))

print(res)
#history_df = pd.DataFrame(history.history)

#ax1 = history_df[['loss', 'val_loss']].plot()

#ax1.set_title("Sigmoid Focal Cross-Entropy")

#ax1.set_xlabel("Epoch #")

#ax1.set_ylabel("Loss")

#ax2 = history_df[['binary_crossentropy', 'val_binary_crossentropy']].plot()

#ax2.set_title("Binary Cross-Entropy")

#ax2.set_xlabel("Epoch #")

#ax2.set_ylabel("Loss")

#history_df[['precision','recall','auc']].plot()