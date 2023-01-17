!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index

!pip install ../input/osic-manual-bbox/efficientnet-1.1.0/efficientnet-1.1.0/ -f ./ --no-index
import os

import gc

import glob

import math

import random

from functools import partial, reduce

from tqdm.notebook import tqdm

import warnings



import pydicom



import scipy

import numpy as np

import pandas as pd



import lightgbm as lgb

import tensorflow as tf

print(f"TF version: {tf.__version__}")

import efficientnet.tfkeras as efn



from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.model_selection import GroupKFold, KFold, train_test_split

from sklearn.metrics import mean_squared_error



import matplotlib.pyplot as plt

import seaborn as sns
import re





def tryint(s):

    try:

        return int(s)

    except ValueError:

        return s

    

def alphanum_key(s):

    """ Turn a string into a list of string and number chunks.

        "z23a" -> ["z", 23, "a"]

    """

    return [ tryint(c) for c in re.split('([0-9]+)', s) ]



def sort_nicely(l):

    """ Sort the given list in the way that humans expect.

    """

    l.sort(key=alphanum_key)



class OSICTrainDataset:

    def __init__(self, df):

        self.df = df

        self.create_base_df()

        self._clean_dataset()

        self._add_base_features()

        self._add_col_id()

        self.__sort_by_id()

        

    def create_base_df(self):

        temp_dff = self.df.copy()

        temp_dff['rank'] = temp_dff.groupby('Patient')['Weeks'].rank(method='min')

        temp_dff = temp_dff[temp_dff['rank'] == 1]

        temp_dff = temp_dff.drop_duplicates(subset='Patient')

        self.base_df = temp_dff



    def _clean_dataset(self):

        """

        Preprocessing steps:

            1. Drop duplicate Patient-Weeks combination

        """

        self.__drop_duplicates()

        

    def __drop_duplicates(self):

        before = self.df.shape[0]

        self.df = self.df.drop_duplicates(subset=['Patient', 'Weeks'], keep='first').reset_index(drop=True)

        after = self.df.shape[0]

        print(f"Dropped {before-after} rows of duplicate 'Patient-Weeks' values.")

        

    def _add_base_features(self):

        before = self.df.shape

        temp_dff = self.df.copy()

        temp_dff['rank'] = temp_dff.groupby('Patient')['Weeks'].rank(method='min')



        all_dfs = []

        for rank in sorted(temp_dff['rank'].unique()):

            all_dfs.append(self.__get_ranked_base_features(temp_dff, rank))

        self.df = pd.concat(all_dfs)

        self.df = self.df.reset_index(drop=True)

        after = self.df.shape

        print(f"Before-After shape adding base features: {before} {after}")

        

    def __get_ranked_base_features(self, temp_dff, rank):

        temp_df = temp_dff[temp_dff['rank'] == rank].reset_index(drop=True)

        temp_df = temp_df.drop(['Sex', 'SmokingStatus', 'rank'], axis=1)

        temp_df = temp_df.rename(columns={

            "FVC": "FVC_base",

            "Percent": "Percent_base",

            "Age": "Age_base",

            "Weeks": "Weeks_base"

        })

        temp_df = self.df[['Patient', 'Weeks', 'FVC', 'Sex', 'SmokingStatus']].merge(

            temp_df,

            how='inner',

            on='Patient',

        )

        temp_df['Weeks_passed'] = temp_df['Weeks'] - temp_df['Weeks_base']

        temp_df = temp_df[temp_df['Weeks_passed'] != 0]

        return temp_df

    

    def _add_col_id(self):

        col_id = 'Patient_Week'

        self.df[col_id] = self.df['Patient'] + '_' + self.df['Weeks'].astype(str)

        print(f"ID column '{col_id}' added. After adding shape: {self.df.shape}")

        

    def __sort_by_id(self):

        self.df = self.df.sort_values(by='Patient').reset_index(drop=True)

        

        

class OSICTestDataset:

    def __init__(self, test_df, submission_df):

        self.df = test_df

        self.submission_df = submission_df

        self._prepare_test_df()

        self.__sort_by_id()

    

    def _prepare_test_df(self):

        before = self.df.shape

        self.submission_df[['Patient', 'Weeks']] = self.submission_df['Patient_Week'].str.split('_', expand=True)

        self.df = self.submission_df.drop(['FVC', 'Confidence'], axis=1).merge(

            self.df.rename(columns={

                "FVC": "FVC_base",

                "Percent": "Percent_base",

                "Age": "Age_base",

                "Weeks": "Weeks_base"

            }),

            how='left',

            on='Patient'

            )

        self.df['Weeks'] = self.df['Weeks'].astype(int)

        self.df['Weeks_passed'] = self.df['Weeks'] - self.df['Weeks_base']

        self.df = self.df.reset_index(drop=True)

        after = self.df.shape

        print(f"Before-After shape adding base features: {before} {after}")

        

    def __sort_by_id(self):

        self.df = self.df.sort_values(by='Patient').reset_index(drop=True)
basepath = "../input/osic-pulmonary-fibrosis-progression/"

train_df = pd.read_csv(f"{basepath}train.csv")

test_df = pd.read_csv(f"{basepath}test.csv")

submission_df = pd.read_csv(f"{basepath}sample_submission.csv")

print(train_df.shape, test_df.shape, submission_df.shape)
train_dataset = OSICTrainDataset(train_df)

test_dataset = OSICTestDataset(test_df, submission_df)
bbox_map = pd.read_csv('../input/osic-manual-bbox/threshold_all.csv')

print(bbox_map.shape)

code_filter = (bbox_map['x'] == 0) & (bbox_map['y'] == 0) & (bbox_map['width'] == 1) & (bbox_map['height'] == 1)

bbox_map = bbox_map[~code_filter]

print(bbox_map.shape)
for x in test_df.Patient:

    print(x in bbox_map.patient.tolist())
import collections



def create_scaler(min_, max_):

    def scalar_scaler(val):

        result = (val - min_) / (max_ - min_)

        result = max(0.0, result)

        result = min(1.0, result)

        return result

    return scalar_scaler



def osic_cat_encoder(cat):

    mapper = {

        'Male': 0,

        'Female': 1,

        'Never smoked': [0, 0],

        'Ex-smoker': [0, 1],

        'Currently smokes': [1, 0],

    }

    return mapper.get(cat, [1, 1])



def flatten(l):

    for el in l:

        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):

            yield from flatten(el)

        else:

            yield el
base_df = train_dataset.df.copy().reset_index(drop=True)

base_test_df = test_dataset.df.copy().reset_index(drop=True)

print(base_df.shape, base_test_df.shape)
BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

base_df = base_df[~base_df.Patient.isin(BAD_ID)]

base_test_df = base_test_df[~base_test_df.Patient.isin(BAD_ID)]

print(base_df.shape, base_test_df.shape)
cols_pred_num = ['FVC_base', 'Percent_base', 'Age_base', 'Weeks_passed']

cols_pred_cat = ['Sex', 'SmokingStatus']

cols_pred = cols_pred_num + cols_pred_cat



for col in cols_pred_num:

    print(f"Predictor {col}")

    min_ = min(base_df[col].min(), test_dataset.df[col].min())

    max_ = max(base_df[col].max(), test_dataset.df[col].max())

    print(min_, max_)

    scaler = create_scaler(min_, max_)

    base_df[col] = base_df[col].apply(scaler)

    base_test_df[col] = base_test_df[col].apply(scaler)

for col in cols_pred_cat:

    base_df[col] = base_df[col].apply(osic_cat_encoder)

    base_test_df[col] = base_test_df[col].apply(osic_cat_encoder)
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler





quantiles = [0.2, 0.5, 0.8]
def score(y_true, y_pred):

    y_true = tf.dtypes.cast(y_true, tf.float32)

    y_pred = tf.dtypes.cast(y_pred, tf.float32)

    C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    

    #sigma_clip = sigma + C1

    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true[:, 0] - fvc_pred)

    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )

    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)

    return -K.mean(metric)



def mloss(_lambda):

    def loss(y_true, y_pred):

        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)

    return loss



def qloss(y_true, y_pred):

    q = tf.constant(quantiles, dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    return K.mean(v)
def effnet_mlp_model(input_shape, output_shape):

    inp = Input(shape=input_shape)

    base_model = efn.EfficientNetB5(

        input_shape=input_shape,

        weights=None,

        include_top=False,

    )

    x1 = base_model(inp)

    x1 = GlobalAveragePooling2D()(x1)

    

    inp2 = Input(shape=(7,))

    x2 = Dense(100, activation='relu')(inp2)

    x2 = Dense(100, activation='relu')(x2)

    

    x = Concatenate()([x1, x2])

    output = Dense(output_shape, activation='linear', name='output')(x)

    

    model = Model(inputs=[inp, inp2], outputs=output, name='cnn_mlp_only_mid')

    return model
model = effnet_mlp_model(input_shape=(512, 512, 1), output_shape=len(quantiles))

model.compile(loss=mloss(0.65),

              optimizer=Adam(learning_rate=0.02),

              metrics=[score])
model.load_weights("../input/osic-pretrained/cnn_mlp_only_mid.h5")
from skimage.filters import threshold_otsu, median

from skimage.segmentation import clear_border

from skimage.transform import resize

from skimage import morphology

from scipy.ndimage import binary_fill_holes





def preprocess_slice(slice, bbox_map, image_type, patient_id, img_width=512, img_height=512):

    slice_data = to_HU(slice)



    slice_data = segment_lung(slice_data, image_type, morphological_segmentation)

    if patient_id not in bbox_map.patient.values:

        bbox = infer_bounding_box(slice_data)

    else:

        x, y, width, height = bbox_map.loc[bbox_map.patient == patient_id, ['x', 'y', 'width', 'height']].values[0]

        bbox = BoundingBox((x, y), width, height)



    slice_data = crop_recenter(slice_data, bbox)

    slice_data = rescale(slice_data)

    resized_slice_data = resize(slice_data, (img_height, img_width), anti_aliasing=True)

    resized_slice_data = np.expand_dims(resized_slice_data, axis=-1)

    resized_slice_data = resized_slice_data.astype(np.float32)

    return resized_slice_data





def segment_lung(slice_data, image_type, segment_func):

    if image_type == 'zero':

        slice_data[slice_data == 0] = -1000

    segmented_image = segment_func(threshold_slices_data(slice_data, low=-1000, high=-400))

    return segmented_image



def morphological_segmentation(img):

    segmented_img, _ = lung_segment(img)

    return segmented_img



def lung_segment(img):

    try:

        thresh = threshold_otsu(img)

        binary = img <= thresh



        lungs = median(clear_border(binary))

        lungs = morphology.binary_closing(lungs, selem=morphology.disk(7))

        lungs = binary_fill_holes(lungs)



        final = lungs*img

        final[final == 0] = np.min(img)

        return final, lungs

    except:

        print("Segmentation failed. Returning original image.")

        return img, img





def infer_bounding_box(segmented_image):

    try:

        y_match, x_match = np.where(segmented_image != -1000)

        y_min, x_min = y_match.min(), x_match.min()

        y_max, x_max = y_match.max(), x_match.max()

        width = abs(x_max - x_min)

        height = abs(y_max - y_min)

    except:

        print("Inferring boundigng box failed, returning whole image")

        x_min = 0

        y_min = 0

        height, width = segmented_image.shape

        height = height - 1

        width = width - 1

    return BoundingBox((x_min, y_min), width, height)



class BoundingBox:

    """Initiation of bbox follows matplotlib Rectangle patch"""

    def __init__(self, xy, width, height):

        self.x, self.y = xy

        self.width = width

        self.height = height

        

    @property

    def attribute_list(self):

        return [(self.x, self.y), self.width, self.height]

    

    def __repr__(self):

        return f"Bbox (bottom left width height): {self.x} {self.y} {self.width} {self.height}"



    

def crop_recenter(image, bbox, pad_value=-1000):

    x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height

    cropped_image = image[ y:y+height, x:x+width ]

    out_height, out_width = image.shape

    

    padded_image = np.ones(image.shape, dtype=np.int16) * pad_value

    x_start = (out_width - width) // 2

    y_start = (out_height - height) // 2

    padded_image[ y_start:y_start+height, x_start:x_start+width ] = cropped_image

    return padded_image





def threshold_slices_data(slices_data, low=-1000, high=-400):

    copy = slices_data.copy()

    copy[copy < low] = low

    copy[copy > high] = high

    return copy





def rescale(slice_data):

    min_, max_ = slice_data.min(), slice_data.max()

    rescaled = (slice_data-min_) / (max_-min_)



    if np.isfinite(rescaled).all():

        return rescaled

    else:

        print("Rescaling failed, returning np.zeros() with original shape.")

        return np.zeros(slice_data.shape)





def to_HU(slice):

    intercept, slope = slice.RescaleIntercept, slice.RescaleSlope



    slice_data = slice.pixel_array.astype(np.int16)

    slice_data[slice_data <= -1000] = 0



    if slope != 1:

        slice_data = slope * slice_data.astype(np.float64)

        slice_data = slice_data.astype(np.int16)



    slice_data += np.int16(intercept)

    return slice_data
DOUBLE_IDS = ['ID00078637202199415319443']
image_mapper = {}

for patient_id in base_test_df.Patient.unique():

    slices = []

    filepaths = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{patient_id}/')

    sort_nicely(filepaths)

    for filepath in filepaths:

        slices.append(pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/test/{patient_id}/{filepath}'))

    if patient_id in DOUBLE_IDS:

        x = slices[(len(slices)-1) //4]

    else:

        x = slices[(len(slices)-1) //2]

    

    intercept = x.RescaleIntercept

    if intercept == 0:

        image_type = 'zero'

    else:

        image_type = 'not-zero'

    x = preprocess_slice(x, bbox_map, image_type, patient_id, 512, 512)

    

    image_mapper.update({

        f'{patient_id}': x,

    })
base_test_df['FVC'] = 2000

base_test_df['Confidence'] = 300

for patient_week in base_test_df['Patient_Week'].unique():

    selector = base_test_df['Patient_Week'] == patient_week

    dff = base_test_df[selector]

    patient_id = dff['Patient'].values[0]

    x = image_mapper[patient_id]

    vector = np.array(list(flatten(dff[cols_pred].values[0])))

    x = np.expand_dims(x, axis=0)

    vector = np.expand_dims(vector, axis=0)

    

    pred = model.predict([x, vector])

    FVC = pred[0][1]

    confidence = pred[0][2] - pred[0][0]

    base_test_df.loc[selector, 'FVC'] = FVC

    base_test_df.loc[selector, 'Confidence'] = confidence
submission_df = submission_df[['Patient_Week']].merge(

    base_test_df[['Patient_Week', 'FVC', 'Confidence']],

    how='inner',

    on='Patient_Week'

)

submission_df.to_csv("submission.csv", header=True, index=False)