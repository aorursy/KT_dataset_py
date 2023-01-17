!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index

!pip install ../input/osic-manual-bbox/efficientnet-1.1.0/efficientnet-1.1.0/ -f ./ --no-index
import os

import gc

import glob

import math

import random

from functools import partial, reduce

from tqdm.auto import tqdm

import warnings



from skimage.transform import resize

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
def plot_slices_data(slices_data, n_cols=10, cmap='gray', **kwargs):

    n_rows = math.ceil(slices_data.shape[0] / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows*1.5))

    for img, ax in tqdm(zip(slices_data, axes.reshape(-1)), leave=False, total=slices_data.shape[0]):

        ax.imshow(img, cmap=cmap, **kwargs)

        ax.axis('off')

    

    missing_image_cnt = (n_rows * n_cols) - slices_data.shape[0]

    if missing_image_cnt > 0:

        for ax in axes.reshape(-1)[::-1][:-missing_image_cnt]:

            ax.axis('off')
def seed_everything(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

#     torch.manual_seed(seed)

#     torch.cuda.manual_seed(seed)

#     torch.backends.cudnn.deterministic = True
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

        

        # Treate every FVC measurement as a baseline

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

        temp_df = temp_df[temp_df['Weeks_passed'] != 0]  # drop base observation

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
class DICOMImages:

    DOUBLE_IDS = ['ID00078637202199415319443']

    """Wrapper for multiple slices of a patient CT-Scan results."""

    def __init__(self, id, dirpath='../input/osic-pulmonary-fibrosis-progression/train/'):

        self.id = id

        self.basepath = os.path.join(dirpath, self.id)

        self.filepaths = glob.glob(os.path.join(self.basepath, "*.dcm"))

        if self.id in self.DOUBLE_IDS:

            self.filepaths = self.filepaths[:len(self.filepaths)//2]

        sort_nicely(self.filepaths)

        

    def __iter__(self):

        for filepath in self.filepaths:

            yield pydicom.dcmread(filepath)



    def __len__(self):

        return len(self.filepaths)

    

    @property

    def image_type(self):

        """

        Infer dicom image type by its first slice metadata.

        Categories:

            - 'zero' : Rescale Intercept value is 0

            - 'not-zero': Rescale Intercept value is either -1000 or -1024

        """

        mapper = {0: 'zero'}

        rescale_intercept = self.get_dicom_metadata(self.get_slice(index=0))['Rescale Intercept']

        return {

            'name': mapper.get(rescale_intercept, 'not-zero'),

            'rescale_intercept': rescale_intercept

        }

        

    @property

    def slices(self):

        return list(self.__iter__())

    

    def get_slice(self, index):

        return pydicom.dcmread(self.filepaths[index])

    

    @property

    def df(self):

        return pd.DataFrame(

            [self.get_dicom_metadata(slice) for slice in self.__iter__()]

        )

    

    @staticmethod

    def get_dicom_metadata(slice):

        dict_res = {}

        for x in slice.values():

            if isinstance(x, pydicom.dataelem.RawDataElement):

                metadata = pydicom.dataelem.DataElement_from_raw(x)

            else:

                metadata = x

            if metadata.name == 'Pixel Data':

                continue

            dict_res.update({

                f"{metadata.name}": metadata.value

            })

        return dict_res

    

    @property

    def slices_data(self):

        return np.stack([self._to_HU(slice) for slice in self.__iter__()])

    

    @property

    def middle_filepath(self):

        return self.filepaths[(len(self.filepaths)-1) // 2]



    @property

    def middle_slice_data(self):

        mid_slice_index = (len(self.filepaths)-1) // 2

        return self._to_HU(pydicom.dcmread(self.filepaths[mid_slice_index]))

        

    def sampled_slices_data(self, n_samples=30, ret_paths=False):

        if len(self.filepaths) < n_samples:

            msg = f"Total slices is less than number of samples: {len(self.filepaths)} < {n_samples}."

            msg += " Number of samples default to total slices."

            warnings.warn(msg, UserWarning)

            n_samples = len(self.filepaths)

        sample_indexes = np.linspace(0, len(self.slices)-1, n_samples).astype(int)

        sampled_slices = np.array(self.slices)[sample_indexes]

        if ret_paths:

            sample_filepaths = np.array(self.filepaths)[sample_indexes]

            return np.stack([self._to_HU(slice) for slice in sampled_slices]), sample_filepaths

        else:

            return np.stack([self._to_HU(slice) for slice in sampled_slices])



    @staticmethod

    def _to_HU(slice):

        intercept, slope = slice.RescaleIntercept, slice.RescaleSlope

        

        slice_data = slice.pixel_array.astype(np.int16)

        slice_data[slice_data <= -1000] = 0

        

        if slope != 1:

            slice_data = slope * slice_data.astype(np.float64)

            slice_data = slice_data.astype(np.int16)

            

        slice_data += np.int16(intercept)

        return slice_data
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
# Ensure our test patient is in our filtered dataset

for x in test_df.Patient:

    print(x in bbox_map.patient.tolist())
def check_bad_images(patient_ids):

    bad_ids = []

    exceptions = []

    pbar = tqdm(range(100), leave=False)  # dummy range

    for patient_id in tqdm(patient_ids, leave=False):

        DICOMImage = DICOMImages(patient_id)

        pbar.reset(total=len(DICOMImage))

        for dicom_image in DICOMImage:

            try:

                _ = dicom_image.pixel_array

            except Exception as e:

                bad_ids.append(patient_id)

                exceptions.append(e)

                break

            finally:

                pbar.update()

        pbar.refresh()

    return bad_ids, exceptions
bad_ids, exceptions = check_bad_images(train_df.Patient.unique())

for bad_id, exception in zip(bad_ids, exceptions):

    print(bad_id, exception)

# bad_ids = ['ID00011637202177653955184', 'ID00052637202186188008618']
def load_image(filename, label=None):

    image = preprocess_dicom(filename)

    if label is None:

        return image

    else:

        return image, label

    

def preprocess_dicom(patient_id, img_width=512, img_height=512):

    middle_slice_data = DICOMImages(patient_id).middle_slice_data

    middle_slice_data = rescale(middle_slice_data)

    middle_slice_data = np.expand_dims(middle_slice_data, axis=-1)

    resized_slice_data = tf.image.resize_with_crop_or_pad(middle_slice_data, img_width, img_height)

    return resized_slice_data

        

def rescale(slice_data):

    min_, max_ = slice_data.min(), slice_data.max()

    rescaled = (slice_data-min_) / (max_-min_)

    total_pixel_count = reduce(lambda a, b: a*b, slice_data.shape)

    assert (rescaled >= 0).sum() == total_pixel_count

    assert (rescaled <= 1).sum() == total_pixel_count

    return rescaled
from skimage.filters import threshold_otsu, median

from skimage.segmentation import clear_border

from skimage import morphology

from scipy.ndimage import binary_fill_holes





def lung_segment(img):

    thresh = threshold_otsu(img)

    binary = img <= thresh



    lungs = median(clear_border(binary))

    lungs = morphology.binary_closing(lungs, selem=morphology.disk(7))

    lungs = binary_fill_holes(lungs)



    final = lungs*img

    final[final == 0] = np.min(img)



    return final, lungs



def morphological_segmentation(img):

    segmented_img, _ = lung_segment(img)

    return segmented_img



def segment_lung(slice_data, image_type, segment_func):

    if image_type == 'zero':

        slice_data[slice_data == 0] = -1000

    segmented_image = segment_func(threshold_slices_data(slice_data, low=-1000, high=-400))

    return segmented_image





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
import collections





def create_scaler(min_, max_):

    def scalar_scaler(val):

        return (val - min_) / (max_ - min_)

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

base_df = base_df[base_df.Patient.isin(bbox_map.patient)]

base_test_df = base_test_df[base_test_df.Patient.isin(bbox_map.patient)]

print(base_df.shape, base_test_df.shape)
cols_pred_num = ['FVC_base', 'Percent_base', 'Age_base', 'Weeks_passed']

cols_pred_cat = ['Sex', 'SmokingStatus']

cols_pred = cols_pred_num + cols_pred_cat



# Apply scaler and cat encoder, look for min-max range

# in both train and test dataset

for col in cols_pred_num:

    min_ = min(base_df[col].min(), test_dataset.df[col].min())

    max_ = max(base_df[col].max(), test_dataset.df[col].max())

    scaler = create_scaler(min_, max_)

    base_df[col] = base_df[col].apply(scaler)

    base_test_df[col] = base_test_df[col].apply(scaler)

for col in cols_pred_cat:

    base_df[col] = base_df[col].apply(osic_cat_encoder)

    base_test_df[col] = base_test_df[col].apply(osic_cat_encoder)
def get_X_y_cnnmlp(df):

    base_df = df.copy()



    mapper = {}

    for patient_id in tqdm(base_df.Patient.unique(), leave=False):

        dicom = DICOMImages(patient_id)

        mapper.update({

            f'{patient_id}': {

                'filepath': dicom.middle_filepath,

                'image_type': dicom.image_type['name'],

            }

        })



    base_df['id'] = base_df['Patient']

    base_df['filepath'] = base_df['Patient'].apply(lambda id: mapper[id]['filepath'])

    base_df['image_type'] = base_df['Patient'].apply(lambda id: mapper[id]['image_type'])

    

    X = base_df[['id', 'filepath', 'image_type']].to_dict(orient='records')

    for x, vector in tqdm(zip(X, base_df[cols_pred].values), leave=False, total=len(X)):

        x.update({

            'vector': list(flatten(vector.tolist()))

        })

    X = np.array(X)

    if 'FVC' in base_df.columns.values:

        y = base_df['FVC'].values

        return X, y

    else:

        return X
# Take only middle slice data

mapper = {}

for patient_id in tqdm(base_df.Patient.unique(), leave=False):

    dicom = DICOMImages(patient_id)

    mapper.update({

        f'{patient_id}': {

            'filepath': dicom.middle_filepath,

            'image_type': dicom.image_type['name'],

        }

    })
base_df['id'] = base_df['Patient']

base_df['filepath'] = base_df['Patient'].apply(lambda id: mapper[id]['filepath'])

base_df['image_type'] = base_df['Patient'].apply(lambda id: mapper[id]['image_type'])
X = base_df[['id', 'filepath', 'image_type']].to_dict(orient='records')

for x, vector in tqdm(zip(X, base_df[cols_pred].values), leave=False, total=len(X)):

    x.update({

        'vector': list(flatten(vector.tolist()))

    })

X = np.array(X)

y = base_df['FVC'].values

print(X.shape, y.shape)
# weeks_scaler = create_scaler(train_df['Weeks'].min(), train_df['Weeks'].max())

# percent_scaler = create_scaler(train_df['Percent'].min(), train_df['Percent'].max())

# age_scaler = create_scaler(train_df['Age'].min(), train_df['Age'].max())

# # fvc_scaler = create_scaler(train_df['FVC'].min(), train_df['FVC'].max())
# base_df = train_dataset.base_df.reset_index(drop=True)

# base_df = base_df[base_df.Patient.isin(bbox_map.patient)]

# base_df.shape
# base_df['Weeks'] = base_df['Weeks'].apply(weeks_scaler)

# base_df['Percent'] = base_df['Percent'].apply(percent_scaler)

# base_df['Age'] = base_df['Age'].apply(age_scaler)

# base_df['Sex'] = base_df['Sex'].apply(osic_cat_encoder)

# base_df['SmokingStatus'] = base_df['SmokingStatus'].apply(osic_cat_encoder)
# all_dicoms = [DICOMImages(patient_id) for patient_id in base_df.Patient.values]

# X = []

# y = []

# for dicom in tqdm(all_dicoms, leave=False):

#     patient_id = dicom.id

#     image_type = dicom.image_type['name']

#     base_data = base_df[base_df['Patient'] == patient_id].values

#     base_FVC = base_data[0, 2]

#     base_PCT = base_data[0, 3]

    

#     sampled_slices_data, sampled_filepaths = dicom.sampled_slices_data(100, ret_paths=True)  # sample or all

#     ratio_from_middle = 0.25 * len(sampled_filepaths)

#     mid_slice_index = len(sampled_filepaths) / 2

#     left = int(mid_slice_index - ratio_from_middle)

#     right = int(mid_slice_index + ratio_from_middle)

#     sampled_filepaths = sampled_filepaths[left:right]

#     total = len(sampled_filepaths)

#     base_data = np.tile(base_data, (total, 1))



#     # Assume 99.7% -- 3 standard deviation. For PCT, since it

#     # is ratio of FVC then the error rate is sqrt(2) of fvc error

#     # assuming error rate of both measurement is 3%

#     FVC_noise = np.random.normal(0, 0.03/3, total) * base_FVC

#     PCT_noise = np.random.normal(0, math.sqrt(2)*0.03/3, total) * base_PCT

# #     FVC_noise = 0

# #     PCT_noise = 0

#     base_data[:, 2] = base_data[:, 2] + FVC_noise

#     base_data[:, 3] = base_data[:, 3] + PCT_noise

    

#     # Change ID to filepaths

#     base_data[:, 0] = sampled_filepaths

    

#     y_index = 2

#     id_index = 0

#     for data in base_data:

#         x_index = list(set(range(7)) - set([y_index, id_index]))

#         x = {

#             'id': patient_id,

#             'filepath': data[0],

#             'image_type': image_type,

#             'vector': list(flatten(data[x_index].tolist())),

#         }

#         X.append(x)

#         y.append(data[y_index])
# gc.collect()
import random

from tensorflow.keras.utils import Sequence





class OSIC_CNNMLP_ImageGenerator(Sequence):

    def __init__(self, vectors, labels, bbox_map, output_shape, batch_size=4, num_batch=0,

                 segmented=True, shuffle=True, debug=False):

        self.vectors = vectors

        self.labels = labels

        self.bbox_map = bbox_map

        self.output_shape = output_shape

        self.batch_size = batch_size

        self.num_batch = num_batch

        self.segmented = segmented

        self.shuffle = shuffle

        self.debug = debug

        self._n = len(self.vectors)

        self.on_epoch_end()



    def __len__(self):

        ct = len(self.vectors) // self.batch_size

        ct += int((len(self.vectors) % self.batch_size)!=0)

        self.num_batch = ct

        return ct

    

    def __getitem__(self, batch_index):

        indexes_in_batch = self.indexes[

            batch_index * self.batch_size:(batch_index + 1) * self.batch_size

        ]



        selected_vectors = self.vectors[indexes_in_batch]

        selected_labels = self.labels[indexes_in_batch]

        out_height, out_width = self.output_shape

        X = np.stack([self.__preprocess_vector(vector, out_width, out_height) for vector in selected_vectors])

        X_meta = np.stack([vector['vector'] for vector in selected_vectors]).astype(np.float32)

        y = np.expand_dims(selected_labels.astype(np.float32), axis=-1)

        if self.debug:

            selected_ids = [v['id'] for v in selected_vectors]

            return [X, X_meta], y, selected_ids

        else:

            return [X, X_meta], y

    

    def on_epoch_end(self):

        self.indexes = np.arange(self._n)

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __preprocess_vector(self, vector, img_width=512, img_height=512):

        slice_data = self.__to_HU(pydicom.dcmread(vector['filepath']))

        image_type = vector['image_type']

        patient_id = vector['id']

        

        if self.segmented:

            try:

                slice_data = segment_lung(slice_data, image_type, morphological_segmentation)

            except:

                print("Segmentation failed, returning original image.")

        x, y, width, height = self.bbox_map.loc[bbox_map.patient == patient_id, ['x', 'y', 'width', 'height']].values[0]

        bbox = BoundingBox((x, y), width, height)

        

        slice_data = crop_recenter(slice_data, bbox)

        slice_data = self.__rescale(slice_data)

        resized_slice_data = resize(slice_data, (img_height, img_width), anti_aliasing=True)

        resized_slice_data = np.expand_dims(resized_slice_data, axis=-1)

        resized_slice_data = resized_slice_data.astype(np.float32)

        return resized_slice_data

    

    def __rescale(self, slice_data):

        min_, max_ = slice_data.min(), slice_data.max()

        rescaled = (slice_data-min_) / (max_-min_)

        

        if np.isfinite(rescaled).all():

            # total_pixel_count = reduce(lambda a, b: a*b, slice_data.shape)

            # assert (rescaled >= 0).sum() == total_pixel_count

            # assert (rescaled <= 1).sum() == total_pixel_count

            return rescaled

        else:

            print("Rescaling failed, returning np.zeros() with original shape.")

            return np.zeros(slice_data.shape)



    

    def __to_HU(self, slice):

        intercept, slope = slice.RescaleIntercept, slice.RescaleSlope

        

        slice_data = slice.pixel_array.astype(np.int16)

        slice_data[slice_data <= -1000] = 0

        

        if slope != 1:

            slice_data = slope * slice_data.astype(np.float64)

            slice_data = slice_data.astype(np.int16)

            

        slice_data += np.int16(intercept)

        return slice_data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train), len(X_valid), len(y_train), len(y_valid))
output_shape = (456, 456)

train_image_dataset = OSIC_CNNMLP_ImageGenerator(

    np.array(X_train), np.array(y_train),

    bbox_map,

    output_shape=output_shape, batch_size=8, segmented=True, shuffle=True, debug=True

)

valid_image_dataset = OSIC_CNNMLP_ImageGenerator(

    np.array(X_valid), np.array(y_valid),

    bbox_map,

    output_shape=output_shape, batch_size=8, segmented=True, shuffle=False

)
for data, labels, ids in tqdm(train_image_dataset, leave=False, total=len(X_train)):

    try:

        images, vector = data

        print(vector.shape)

        plot_slices_data(images.reshape(images.shape[0], 456, 456), cmap='Blues_r', n_cols=8)

    except:

        print(ids)

    break
class ModelExtractionCallback(object):

    """Callback class for retrieving trained model from lightgbm.cv()

    NOTE: This class depends on '_CVBooster' which is hidden class,

    so it might doesn't work if the specification is changed.

    """



    def __init__(self):

        self._model = None



    def __call__(self, env):

        # Saving _CVBooster object.

        self._model = env.model



    def _assert_called_cb(self):

        if self._model is None:

            # Throw exception if the callback class is not called.

            raise RuntimeError('callback has not called yet')



    @property

    def boosters_proxy(self):

        self._assert_called_cb()

        # return Booster object

        return self._model



    @property

    def raw_boosters(self):

        self._assert_called_cb()

        # return list of Booster

        return self._model.boosters



    @property

    def best_iteration(self):

        self._assert_called_cb()

        # return boosting round when early stopping.

        return self._model.best_iteration



def get_proxy_boosters_best_iter(extraction_cb):

    proxy = extraction_cb.boosters_proxy

    boosters = extraction_cb.raw_boosters

    best_iteration = extraction_cb.best_iteration

    return proxy, boosters, best_iteration



def loss_func(y_true, y_pred, weight):

    confidence = weight

    sigma_clipped = max(confidence, 70)

    diff = abs(y_true - y_pred)

    delta = min(diff, 1000)

    score = -math.sqrt(2)*delta/sigma_clipped - np.log(math.sqrt(2)*sigma_clipped)

    return -score
"""GLOBAL CONFIGS"""

SEED = 42

seed_everything(SEED)



cols_num = ['FVC_base', 'Percent_base', 'Age_base', 'Weeks_passed']

cols_cat = ['Sex', 'SmokingStatus']

cols_cat_oe = [c+'_oe' for c in cols_cat]

cols_pred = cols_num + cols_cat_oe



col_target = 'FVC'

col_target_2 = 'Confidence'

col_score = 'Score'
lgbm_param = {

    'objective': 'regression',

    'boosting': 'gbdt',

    'metric': 'rmse',

    'learning_rate': 0.01,

    'num_leaves': 31,

    'max_depth': 2,

    'colsample_bytree': 0.8,

    'subsample': 0.8,

    'subsample_freq': 1,

    'num_threads': os.cpu_count() - 1,

    'seed': 42

}



lgbm_train_param = {

    "verbose_eval": 100,

    "num_boost_round": 100000,

    "early_stopping_rounds": 100,

}
train_oe = OrdinalEncoder()

train_dataset.df[cols_cat_oe] = train_oe.fit_transform(train_dataset.df[cols_cat])

test_dataset.df[cols_cat_oe] = train_oe.fit_transform(test_dataset.df[cols_cat])



train_set = lgb.Dataset(train_dataset.df[cols_pred],

                        label=train_dataset.df[col_target],

                        group=train_dataset.df['Patient'].value_counts().sort_index())
extraction_cb = ModelExtractionCallback()



bst = lgb.cv(

    lgbm_param,

    train_set,

    **lgbm_train_param,

    folds = GroupKFold(n_splits=5),

    seed=SEED,

    callbacks=[extraction_cb]

)
proxy, boosters, best_iteration = get_proxy_boosters_best_iter(extraction_cb)

predictions = proxy.predict(train_dataset.df[cols_pred], num_iteration=best_iteration)

for i, preds in enumerate(predictions):

    rmse = np.sqrt(mean_squared_error(train_dataset.df[col_target], preds))

    print(f"Fold {i} rmse: {rmse}")
booster_id = 0

fig, ax = plt.subplots(1, 2, figsize=(14, 4))

lgb.plot_importance(boosters[booster_id], importance_type='gain', ax=ax[0])

lgb.plot_importance(boosters[booster_id], importance_type='split', ax=ax[1])

plt.tight_layout()

plt.show()
def score_dataset(df):

    scores = []

    rows = df[['FVC', 'FVC_pred', 'Confidence']].values

    for y_true, y_pred, weight in rows:

        score = loss_func(y_true, y_pred, weight)

        scores.append(score)

    df[col_score] = scores

    return -np.mean(scores)
# Predict FVC

train_dataset.df['FVC_pred'] = np.array(predictions).mean(axis=0)

test_dataset.df['FVC_pred'] = np.array(proxy.predict(test_dataset.df[cols_pred], num_iteration=best_iteration)).mean(axis=0)



# Optimize score

train_dataset.df['Confidence'] = 100

non_optimized_score = score_dataset(train_dataset.df)



results = []

weight_arr = [100]

for y_true, y_pred in tqdm(train_dataset.df[['FVC', 'FVC_pred']].values, leave=False):

    loss_partial = partial(loss_func, y_true, y_pred)

    result = scipy.optimize.minimize(loss_partial, weight_arr, method='SLSQP')

    x = result['x']

    results.append(x[0])



train_dataset.df['Confidence'] = results

optimized_score = score_dataset(train_dataset.df)

print(f"Non optimized score: {non_optimized_score}")

print(f"Optimized score: {optimized_score}")
cols_pred_2 = list(set(cols_pred) - set(['FVC_base']))

train_set = lgb.Dataset(train_dataset.df[cols_pred_2],

                        label=train_dataset.df[col_target_2],

                        group=train_dataset.df['Patient'].value_counts().sort_index())
extraction_cb = ModelExtractionCallback()



bst = lgb.cv(

    lgbm_param,

    train_set,

    **lgbm_train_param,

    folds = GroupKFold(n_splits=5),

    seed=SEED,

    callbacks=[extraction_cb]

)
proxy, boosters, best_iteration = get_proxy_boosters_best_iter(extraction_cb)

predictions = proxy.predict(train_dataset.df[cols_pred_2], num_iteration=best_iteration)

for i, preds in enumerate(predictions):

    rmse = np.sqrt(mean_squared_error(train_dataset.df[col_target_2], preds))

    print(f"Fold {i} rmse: {rmse}")
# Predict Confidence

train_dataset.df[col_target_2] = np.array(predictions).mean(axis=0)

test_dataset.df[col_target_2] = np.array(proxy.predict(test_dataset.df[cols_pred_2], num_iteration=best_iteration)).mean(axis=0)

print(f"Train score: {score_dataset(train_dataset.df)}")
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
def efficientnet_plus_model(input_shape, output_shape):

    inp = Input(shape=input_shape)

#     inp_concat = Concatenate()([inp, inp, inp])

    base_model = efn.EfficientNetB5(

        input_shape=input_shape,

        weights=None,

        include_top=False,

    )

#     base_model.load_weights("../input/efficientnet/efficientnet-b5_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5")

    x1 = base_model(inp)

    x1 = GlobalAveragePooling2D()(x1)

    

    inp2 = Input(shape=(6,))

    

    x = Concatenate()([x1, inp2])

    output = Dense(output_shape, activation='linear', name='output')(x)

    model = Model(inputs=[inp, inp2],

                  outputs=output,

                  name='effnetb5_plus_osic')

    return model
def efficientnet_model(input_shape, output_shape):

    """

    Although input shape can be modified, I prefer to use

    efficientnet native input shape from model.

    """

    inp = Input(shape=input_shape)

#     inp_concat = Concatenate()([inp, inp, inp])

    base_model = efn.EfficientNetB5(

        input_shape=input_shape,

        weights=None, 

        include_top=False,

    )

#     base_model.load_weights("../input/efficientnet/efficientnet-b5_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5")

    x = base_model(inp)

    x = GlobalAveragePooling2D()(x)

    output = Dense(output_shape, activation='linear', name='output')(x)

    model = Model(inputs=inp, outputs=output, name='effnetb5_osic')

    

    # On-off trainable layers

#     model.layers[0].trainable = False



    return model
def dense_plus_model1(input_shape, output_shape):

    """

    Multiple quantile regression basic dense only model for image input.

    

    Parameters

    ----------

    input_shape : tuple

        except input to be in grayscale

    output_shape : int

        how many quantile to fit

        

    Returns

    -------

    model : tf.keras.models.Model

    """

    input = Input(shape=input_shape, name="input")

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input)

    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu', name='dense1')(x)

    x = Dense(128, activation='relu', name='dense2')(x)

    x = Dense(64, activation='relu', name='dense3')(x)

    x = Dense(16, activation='relu', name='dense4')(x)

    

    input2 = Input(shape=(6,))

    x = Concatenate()([x, input2])

    output = Dense(output_shape, activation='linear', name='output')(x)

    

    model = Model(inputs=[input, input2], outputs=output, name='dense_plus_model1')

    return model
def dense_model1(input_shape, output_shape):

    """

    Multiple quantile regression basic dense only model for image input.

    

    Parameters

    ----------

    input_shape : tuple

        except input to be in grayscale

    output_shape : int

        how many quantile to fit

        

    Returns

    -------

    model : tf.keras.models.Model

    """

    input = Input(shape=input_shape, name="input")

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input)

    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu', name='dense1')(x)

    x = Dense(128, activation='relu', name='dense2')(x)

    x = Dense(64, activation='relu', name='dense3')(x)

    x = Dense(16, activation='relu', name='dense4')(x)

    output = Dense(output_shape, activation='linear', name='output')(x)

    

    model = Model(inputs=input, outputs=output, name='dense_model1')

    return model
# Data Generator

output_shape = (512, 512)

# train_image_dataset = OSICImageGenerator(X_train, y_train, bbox_map, output_shape=output_shape, segmented=True, shuffle=True, debug=False)

# valid_image_dataset = OSICImageGenerator(X_valid, y_valid, bbox_map, output_shape=output_shape, segmented=True, shuffle=False)



train_image_dataset = OSIC_CNNMLP_ImageGenerator(

    np.array(X_train), np.array(y_train),

    bbox_map,

    output_shape=output_shape, batch_size=4, segmented=True, shuffle=True, debug=False

)

valid_image_dataset = OSIC_CNNMLP_ImageGenerator(

    np.array(X_valid), np.array(y_valid),

    bbox_map,

    output_shape=output_shape, batch_size=4, segmented=True, shuffle=False

)
# Learning Rate Scheduler

LR_START = 1e-3

LR_MAX = 0.03

LR_RAMPUP_EPOCHS = 5

LR_SUSTAIN_EPOCHS = 0

LR_STEP_DECAY = 0.75



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = LR_MAX * LR_STEP_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//10)

    return lr

    

lr2 = LearningRateScheduler(lrfn, verbose = True)



rng = [i for i in range(100)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y); 

plt.xlabel('epoch',size=14); plt.ylabel('learning rate',size=14)

plt.title('Training Schedule',size=16); plt.show()
# model = dense_model1(input_shape=(512, 512, 1), output_shape=len(quantiles))

# model = dense_plus_model1(input_shape=(456, 456, 1), output_shape=len(quantiles))

# model = efficientnet_model(input_shape=(456, 456, 1), output_shape=len(quantiles))

# model = efficientnet_plus_model(input_shape=(456, 456, 1), output_shape=len(quantiles))

model = effnet_mlp_model(input_shape=(512, 512, 1), output_shape=len(quantiles))

model.compile(loss=mloss(0.65),

              optimizer=Adam(learning_rate=0.03),

              metrics=[score])
model.summary()
# model_path = "dense_model1.h5"

# model_path = "dense_plus_model1.h5"

# model_path = "effnetb5.h5"

# model_path = "effnetb5_plus.h5"

model_path = "cnn_mlp_only_mid.h5"



checkpoint_path = os.path.abspath(model_path)



cp_callback = ModelCheckpoint(filepath=checkpoint_path,

                              monitor='val_loss',

                              save_best_only=True,

                              mode='min',

                              save_weights_only=True),

es_callback = EarlyStopping(monitor='val_loss',

                            min_delta=1e-4,

                            patience=4,

                            mode='min'),

lr_callback = lr2
history = model.fit(

    train_image_dataset,

    validation_data=valid_image_dataset,

    initial_epoch=0,

    callbacks=[

        cp_callback,

        es_callback,

        lr_callback

    ],

    epochs=1  # estimated ~30 min per epoch, so maximum runtime of 6 hours equals 12 epoch, here I just use 50%

)
valid_score = score(np.expand_dims(y_valid, axis=-1), model.predict(valid_image_dataset))

print(f"Validation score: {valid_score}")
submission_df = pd.read_csv(f"{basepath}sample_submission.csv")

submission_df = submission_df[['Patient_Week']].merge(

    test_dataset.df[['Patient_Week', 'FVC_pred', 'Confidence']]\

        .rename(columns={"FVC_pred": "FVC"}),

    how='inner',

    on='Patient_Week'

)

print(submission_df.head(5))

submission_df.to_csv("submission_lgbm.csv", header=True, index=False)
cols_pred_num = ['FVC_base', 'Percent_base', 'Age_base', 'Weeks_passed']

cols_pred_cat = ['Sex', 'SmokingStatus']

cols_pred = cols_pred_num + cols_pred_cat
X_test = get_X_y_cnnmlp(base_test_df)

test_image_dataset = OSIC_CNNMLP_ImageGenerator(

    np.array(X_test), np.array([2000]*len(X_test)),  # dummy target data

    bbox_map,

    output_shape=output_shape, batch_size=4, segmented=True, shuffle=False

)



y_pred = model.predict(test_image_dataset)
base_test_df['FVC'] = y_pred[:, 1]

base_test_df['Confidence'] = y_pred[:, 2] - y_pred[:, 0]
submission_df = pd.read_csv(f"{basepath}sample_submission.csv")

submission_df = submission_df[['Patient_Week']].merge(

    base_test_df[['Patient_Week', 'FVC', 'Confidence']],

    how='inner',

    on='Patient_Week'

)

print(submission_df.head(5))

submission_df.to_csv("submission_cnn_mlp.csv", header=True, index=False)