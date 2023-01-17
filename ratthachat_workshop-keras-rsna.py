EPOCHS = 0 # แก้จำนวน Epochs ที่จะเทรน โดยค่าสูงสุดที่สามารถรันได้ใน Kaggle Notebook (จำกัดชั่วโมงในการรันไม่เกิน 9 ชั่วโมง) คือ EPOCHS = 5
import numpy as np

import pandas as pd

import pydicom

import os

import matplotlib.pyplot as plt

import collections

from tqdm import tqdm_notebook as tqdm

from datetime import datetime



from math import ceil, floor, log

import cv2



import tensorflow as tf

import keras



import sys



# from keras_applications.resnet import ResNet50

from keras_applications.inception_v3 import InceptionV3



from sklearn.model_selection import ShuffleSplit



!ls ../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection
test_images_dir = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_test/'

train_images_dir = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/'
!ls {train_images_dir} | wc # นับจำนวนรูปใน training data
def window_image(dcm, window_center, window_width):

    

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    img = np.clip(img, img_min, img_max)



    return img



def bsb_window(dcm):

    brain_img = window_image(dcm, 40, 80)

    subdural_img = window_image(dcm, 80, 200)

    soft_img = window_image(dcm, 40, 380)

    

    brain_img = (brain_img - 0) / 80

    subdural_img = (subdural_img - (-20)) / 200

    soft_img = (soft_img - (-150)) / 380 # soft-tissue window

    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)



    return bsb_img
def _read(path, desired_size=(256, 256)):

    """Will be used in DataGenerator"""

    

    dcm = pydicom.dcmread(path)

    

    try:

        img = bsb_window(dcm)

    except:

        img = np.zeros(desired_size)

    

    

    img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)

    

    return img



# ทดสอบโชว์รูปใน 3 windows

# ลองเปลี่ยนชื่อไฟล์เหล่านี้ดู: ID_2669954a7, ID_5c8b5d701, ID_52c9913b1



plt.imshow(

    _read(train_images_dir+'ID_5c8b5d701'+'.dcm', (256, 256)), cmap=plt.cm.bone

);



# ในรูปนี้มีปรากฏการณ์เลือดคั่งในสมอง 3 รูปแบบด้วยกัน เพื่อนๆ มองออกไหมครับ ?

#                                     ID  Label

# 4045566          ID_5c8b5d701_epidural      0

# 4045567  ID_5c8b5d701_intraparenchymal      1

# 4045568  ID_5c8b5d701_intraventricular      0

# 4045569      ID_5c8b5d701_subarachnoid      1

# 4045570          ID_5c8b5d701_subdural      1

# 4045571               ID_5c8b5d701_any      1
class DataGenerator(keras.utils.Sequence):



    def __init__(self, list_IDs, labels=None, batch_size=1, img_size=(512, 512, 1), 

                 img_dir=train_images_dir, *args, **kwargs):



        self.list_IDs = list_IDs

        self.labels = labels

        self.batch_size = batch_size

        self.img_size = img_size

        self.img_dir = img_dir

        self.on_epoch_end()



    def __len__(self):

        return int(ceil(len(self.indices) / self.batch_size))



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



    def __data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.img_size))

        

        if self.labels is not None: # training phase

            Y = np.empty((self.batch_size, 6), dtype=np.float32)

        

            for i, ID in enumerate(list_IDs_temp):

                X[i,] = _read(self.img_dir+ID+".dcm", self.img_size)

                Y[i,] = self.labels.loc[ID].values

        

            return X, Y

        

        else: # test phase

            for i, ID in enumerate(list_IDs_temp):

                X[i,] = _read(self.img_dir+ID+".dcm", self.img_size)

            

            return X
from keras import backend as K



def weighted_log_loss(y_true, y_pred):

    """

    Can be used as the loss function in model.compile()

    ---------------------------------------------------

    """

    

    class_weights = np.array([2., 1., 1., 1., 1., 1.])

    

    eps = K.epsilon()

    

    y_pred = K.clip(y_pred, eps, 1.0-eps)



    out = -(         y_true  * K.log(      y_pred) * class_weights

            + (1.0 - y_true) * K.log(1.0 - y_pred) * class_weights)

    

    return K.mean(out, axis=-1)





def _normalized_weighted_average(arr, weights=None):

    """

    A simple Keras implementation that mimics that of 

    numpy.average(), specifically for this competition

    """

    

    if weights is not None:

        scl = K.sum(weights)

        weights = K.expand_dims(weights, axis=1)

        return K.sum(K.dot(arr, weights), axis=1) / scl

    return K.mean(arr, axis=1)





def weighted_loss(y_true, y_pred):

    """

    Will be used as the metric in model.compile()

    ---------------------------------------------

    

    Similar to the custom loss function 'weighted_log_loss()' above

    but with normalized weights, which should be very similar 

    to the official competition metric:

        https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring

    and hence:

        sklearn.metrics.log_loss with sample weights

    """

    

    class_weights = K.variable([2., 1., 1., 1., 1., 1.])

    

    eps = K.epsilon()

    

    y_pred = K.clip(y_pred, eps, 1.0-eps)



    loss = -(        y_true  * K.log(      y_pred)

            + (1.0 - y_true) * K.log(1.0 - y_pred))

    

    loss_samples = _normalized_weighted_average(loss, class_weights)

    

    return K.mean(loss_samples)





def weighted_log_loss_metric(trues, preds):

    """

    Will be used to calculate the log loss 

    of the validation set in PredictionCheckpoint()

    ------------------------------------------

    """

    class_weights = [2., 1., 1., 1., 1., 1.]

    

    epsilon = 1e-7

    

    preds = np.clip(preds, epsilon, 1-epsilon)

    loss = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)

    loss_samples = np.average(loss, axis=1, weights=class_weights)



    return - loss_samples.mean()





class PredictionCheckpoint(keras.callbacks.Callback):

    

    def __init__(self, test_df, valid_df, 

                 test_images_dir=test_images_dir, 

                 valid_images_dir=train_images_dir, 

                 batch_size=32, input_size=(224, 224, 3)):

        

        self.test_df = test_df

        self.valid_df = valid_df

        self.test_images_dir = test_images_dir

        self.valid_images_dir = valid_images_dir

        self.batch_size = batch_size

        self.input_size = input_size

        

    def on_train_begin(self, logs={}):

        self.test_predictions = []

        self.valid_predictions = []

        

    def on_epoch_end(self,batch, logs={}):

        print('ทำนาย test data...')

        self.test_predictions.append(

            self.model.predict_generator(

                DataGenerator(self.test_df.index, None, self.batch_size, self.input_size, self.test_images_dir), verbose=1)[:len(self.test_df)])



class MyDeepModel:

    

    def __init__(self, engine, input_dims, batch_size=5, num_epochs=4, learning_rate=1e-3, 

                 decay_rate=1.0, decay_steps=1, weights="imagenet", verbose=1):

        

        self.engine = engine

        self.input_dims = input_dims

        self.batch_size = batch_size

        self.num_epochs = num_epochs

        self.learning_rate = learning_rate

        self.decay_rate = decay_rate

        self.decay_steps = decay_steps

        self.weights = weights

        self.verbose = verbose

        self._build()



    def _build(self):

        

        

        engine = self.engine(include_top=False, weights=self.weights, input_shape=self.input_dims,

                             backend = keras.backend, layers = keras.layers,

                             models = keras.models, utils = keras.utils)

        

        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)

        out = keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)



        self.model = keras.models.Model(inputs=engine.input, outputs=out)



        self.model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=[weighted_loss])

    



    def fit_and_predict(self, train_df, valid_df, test_df):

        

        # callbacks

        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)

        scheduler = keras.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))

        

        self.model.fit_generator(

            DataGenerator(

                train_df.index, 

                train_df, 

                self.batch_size, 

                self.input_dims, 

                train_images_dir

            ),

            epochs=self.num_epochs,

            verbose=self.verbose,

            use_multiprocessing=True,

            workers=4,

            callbacks=[pred_history, scheduler]

        )

        

        return pred_history

    

    def save(self, path):

        self.model.save_weights(path)

    

    def load(self, path):

        self.model.load_weights(path)
def read_testset(filename="../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv"):

    df = pd.read_csv(filename)

    df["Image"] = df["ID"].str.slice(stop=12)

    df["Diagnosis"] = df["ID"].str.slice(start=13)

    

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]

    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    

    return df



def read_trainset(filename="../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv"):

    df = pd.read_csv(filename)

    df["Image"] = df["ID"].str.slice(stop=12)

    df["Diagnosis"] = df["ID"].str.slice(start=13)

    

    # เอารูปภาพซ้ำออก

    duplicates_to_remove = [56340, 56341, 56342, 56343, 56344, 56345, 56346, 56347, 56348, 56349, 56350, 56351, 

                            1171824, 1171825, 1171826, 1171827, 1171828, 1171829, 1171830, 1171831, 1171832, 1171833, 

                            1171834, 1171835, 3705306, 3705307, 3705308, 3705309, 3705310, 3705311, 3705312, 3705313, 

                            3705314, 3705315, 3705316, 3705317, 3842472, 3842473, 3842474, 3842475, 3842476, 3842477, 

                            3842478, 3842479, 3842480, 3842481, 3842482, 3842483]

    

    df = df.drop(index=duplicates_to_remove)

    df = df.reset_index(drop=True)

    

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]

    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    

    return df



df = read_trainset()

test_df = read_testset()
# ตัวอย่าง meta-data ของ training set ที่เราต้องการ

df.head(3)
# ตัวอย่าง meta-data ของ test set ซึ่งเราจะเปลี่ยนค่า 0.5 ในตาราง ให้เป็นคำนายเลือดคั่งในสมองจากโมเดลของเราภายหลัง

test_df.head(3)
# train set (00%) and validation set (10%)

ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42).split(df.index)



# lets go for the first fold only

train_idx, valid_idx = next(ss)

# EPOCHS=0

if EPOCHS == 0: # just for illustration on CPU

    train_idx = train_idx[:100]

    valid_idx = valid_idx[:100]

    EPOCHS = 1



# obtain model

model = MyDeepModel(engine=InceptionV3, input_dims=(256, 256, 3), batch_size=32, learning_rate=5e-4,

                    num_epochs=EPOCHS, decay_rate=0.8, decay_steps=1, weights="imagenet", verbose=1)



# obtain test + validation predictions (history.test_predictions, history.valid_predictions)

history = model.fit_and_predict(df.iloc[train_idx], df.iloc[valid_idx], test_df)

test_df = read_testset()



print(history.test_predictions)



test_df.iloc[:, :] = np.mean(history.test_predictions, axis=0)



test_df = test_df.stack().reset_index()



test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])



test_df = test_df.drop(["Image", "Diagnosis"], axis=1)



test_df.to_csv('submission.csv', index=False)

test_df.head()