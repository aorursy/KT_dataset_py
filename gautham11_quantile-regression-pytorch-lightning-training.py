import pandas as pd

import numpy as np

import os

import sys

import random

import torch

import matplotlib.pyplot as plt

from pathlib import Path

import math

import plotly

import plotly_express as px

import seaborn as sns

import itertools

from functools import partial

import tqdm.notebook as tqdm

from scipy.stats import kurtosis, skew

import time

import pydicom

import scipy

import scipy.ndimage as ndimage

from skimage import measure, morphology, segmentation

from scipy.ndimage.interpolation import zoom

from PIL import Image 



# from pytorch_tabnet.tab_network import TabNet



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')







%load_ext autoreload

%autoreload 2
import sys

sys.path.append('../input/pytorchtabnet/tabnet-develop')

from pytorch_tabnet.tab_network import TabNet
!pip install ../input/pytorchlightning/tensorboard-2.2.0-py3-none-any.whl -q

!pip install ../input/pytorchlightning/pytorch_lightning-0.9.0-py3-none-any.whl -q
## Code from https://www.kaggle.com/aadhavvignesh/lung-segmentation-by-marker-controlled-watershed 



def generate_markers(image):

    """

    Generates markers for a given image.

    

    Parameters: image

    

    Returns: Internal Marker, External Marker, Watershed Marker

    """

    

    #Creation of the internal Marker

    marker_internal = image < -400

    marker_internal = segmentation.clear_border(marker_internal)

    marker_internal_labels = measure.label(marker_internal)

    

    areas = [r.area for r in measure.regionprops(marker_internal_labels)]

    areas.sort()

    

    if len(areas) > 2:

        for region in measure.regionprops(marker_internal_labels):

            if region.area < areas[-2]:

                for coordinates in region.coords:                

                       marker_internal_labels[coordinates[0], coordinates[1]] = 0

    

    marker_internal = marker_internal_labels > 0

    

    # Creation of the External Marker

    external_a = ndimage.binary_dilation(marker_internal, iterations=10)

    external_b = ndimage.binary_dilation(marker_internal, iterations=55)

    marker_external = external_b ^ external_a

    

    # Creation of the Watershed Marker

    marker_watershed = np.zeros(image.shape, dtype=np.int)

    marker_watershed += marker_internal * 255

    marker_watershed += marker_external * 128

    

    return marker_internal, marker_external, marker_watershed





def seperate_lungs(image,iterations = 1):

    """

    Segments lungs using various techniques.

    

    Parameters: image (Scan image), iterations (more iterations, more accurate mask)

    

    Returns: 

        - Segmented Lung

        - Lung Filter

        - Outline Lung

        - Watershed Lung

        - Sobel Gradient

    """

    

    # Store the start time

    # start = time.time()

    marker_internal, marker_external, marker_watershed = generate_markers(image)

    

    

    '''

    Creation of Sobel Gradient

    '''

    

    # Sobel-Gradient

    sobel_filtered_dx = ndimage.sobel(image, 1)

    sobel_filtered_dy = ndimage.sobel(image, 0)

    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)

    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    

    

    '''

    Using the watershed algorithm

    

    

    We pass the image convoluted by sobel operator and the watershed marker

    to morphology.watershed and get a matrix matrix labeled using the 

    watershed segmentation algorithm.

    '''

    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    

    '''

    Reducing the image to outlines after Watershed algorithm

    '''

    outline = ndimage.morphological_gradient(watershed, size=(3,3))

    outline = outline.astype(bool)

    

    

    '''

    Black Top-hat Morphology:

    

    The black top hat of an image is defined as its morphological closing

    minus the original image. This operation returns the dark spots of the

    image that are smaller than the structuring element. Note that dark 

    spots in the original image are bright spots after the black top hat.

    '''

    

    # Structuring element used for the filter

    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],

                       [0, 1, 1, 1, 1, 1, 0],

                       [1, 1, 1, 1, 1, 1, 1],

                       [1, 1, 1, 1, 1, 1, 1],

                       [1, 1, 1, 1, 1, 1, 1],

                       [0, 1, 1, 1, 1, 1, 0],

                       [0, 0, 1, 1, 1, 0, 0]]

    

    blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)

    

    # Perform Black Top-hat filter

    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    

    '''

    Generate lung filter using internal marker and outline.

    '''

    lungfilter = np.bitwise_or(marker_internal, outline)

    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)

    

    '''

    Segment lung using lungfilter and the image.

    '''

    segmented = np.where(lungfilter, image, -1000)

    

    #return segmented, lungfilter, outline, watershed, sobel_gradient

    return segmented
def load_scan(path):

    paths = os.listdir(path)

    paths = sorted(paths, key=lambda x: int(str(x).split('/')[-1].split('.')[0]))

    slices = [pydicom.read_file(path + '/' + s) for s in paths]

    print(len(slices), 'slices')

    try:

        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    except:

        pass

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:

        s.SliceThickness = slice_thickness

    return slices



def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    

    

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slice_number] = image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

    image[image <= -1900] = -1000

    return np.array(image, dtype=np.int16)



def resample(image, scan, new_spacing=[1,1,1]):

    st = time.time()

    slice_thickness = scan[0].SliceThickness

    spacing = np.array([slice_thickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    print('resample factor', time.time() - st, real_resize_factor)

    image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')

    print('resample time', time.time() - st)

    return image, new_spacing



def torch_resample(image, scan, new_spacing=[1,1,1]):

    st = time.time()

    slice_thickness = scan[0].SliceThickness

    spacing = np.array([slice_thickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    #print('resample factor', time.time() - st, real_resize_factor)

    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)

    #print(real_resize_factor)

    #size = [int(x*y) for x, y in zip(real_resize_factor, image.shape[-3:])]

    #print(image.shape)

    #print('using resize')

    image = torch.nn.functional.interpolate(image, scale_factor=tuple(real_resize_factor), mode='nearest')

    image = image.squeeze(0).squeeze(0).numpy()

    #print(image.shape)

    #print('resample time', time.time() - st)

    return image, new_spacing









def get_3d_resampled_array(patient_path):

    start_time = time.time()

    patient_slices = load_scan(str(patient_path))

    patient_slices_hu = get_pixels_hu(patient_slices)

    print('HU loaded', time.time() - start_time)

    print(patient_slices_hu.shape)

    #lungmask_3d = np.apply_over_axes(seperate_lungs, patient_slices_hu, 0)

    idx = np.ndindex(patient_slices_hu.shape[0])

    patient_slices_hu_masked = np.zeros(patient_slices_hu.shape)

    for i in idx:

        patient_slices_hu_masked[i] = seperate_lungs(patient_slices_hu[i])

        #patient_slices_hu_masked[i, :, :] = np.where(lungmask, patient_slices_hu[i, :, :], -1000)

    #patient_slices_hu_masked = np.where(lungmask_3d, patient_slices_hu, -1000)

    

    print('mask generated', time.time() - start_time)

    resampled_array, spacing = torch_resample(patient_slices_hu_masked, patient_slices, [1,1,1])

    print('after resample', time.time() - start_time)

    return resampled_array, spacing





def get_features_from_3d_array(resampled_array, spacing):

    features = {}

    

    # volume of lungs

    cube_volume = spacing[0] * spacing[1] * spacing[2]

    total_lung_volume = (resampled_array[resampled_array > -900].shape[0] * cube_volume)

    lung_volume_in_liters = total_lung_volume / (1000*1000)

    features['lung_volume_in_liters'] = lung_volume_in_liters

    

    #HU unit binning

    bins_threshold = (resampled_array <= 300) & (resampled_array >= -900)

    total_hu_units_bin = resampled_array[bins_threshold].flatten().shape[0]

    bin_values, bins = np.histogram(resampled_array[bins_threshold].flatten(), bins=range(-900, 400, 100))

    features['total_hu_units_bin'] = total_hu_units_bin

    for i, _bin in enumerate(bins[:-1]):

        features[f'bin_{_bin}'] = bin_values[i] / total_hu_units_bin

    

    #mean, skew, kurtosis

    lung_threshold = (resampled_array <= -320) & (resampled_array >= -900)

    histogram_values, _ = np.histogram(resampled_array[lung_threshold].flatten(), bins=100)

    features['lung_mean_hu'] = np.mean(resampled_array[lung_threshold].flatten())

    features['lung_skew'] = skew(histogram_values)

    features['lung_kurtosis'] = kurtosis(histogram_values)

    

    #height_of_lung

    n_lung_pixels = lung_threshold.sum(axis=1).sum(axis=1)

    height_start = np.argwhere(n_lung_pixels > 1000).min()

    height_end = np.argwhere(n_lung_pixels > 1000).max()

    features['height_of_lung_cm'] = (height_end - height_start)/10

    

    return features
data_dir = Path('../input/osic-pulmonary-fibrosis-progression')
def seed_everything(seed=42):

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False
seed_everything()
def process_submission(submission):

    submission['Weeks'] = submission['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

    submission['Patient'] = submission['Patient_Week'].apply(lambda x: x.split('_')[0])

    return submission
train = pd.read_csv(data_dir/'train.csv')

test = pd.read_csv(data_dir/'test.csv')

# dicom_meta = pd.read_pickle(data_dir/'train_dicom_df')/

submission = process_submission(pd.read_csv(data_dir/'sample_submission.csv'))

image_feature_df = pd.read_csv('../input/lung-image-features/patient_feature2_df.csv')
image_feature_df =image_feature_df.drop(['Unnamed: 0'], axis=1)
image_feature_df.head()
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold

from sklearn.preprocessing import LabelEncoder



def frequency_binning(x, nbin):

    nlen = len(x)

    return np.unique(np.interp(np.linspace(0, nlen, nbin + 1),

                     np.arange(nlen),

                     np.sort(x)))



def replace_with_frequency_bins(arr, nbins=5):

    bins = frequency_binning(arr, nbins)

    out = pd.cut(arr, bins)

    return [inter.left if (type(inter) is not float) else inter for inter in out]

train['expected_fvc'] = np.round((100 * train['FVC']) / train['Percent'], 2)

test['expected_fvc'] = np.round((100 * test['FVC']) / test['Percent'], 2)

patient_df = train[['Patient', 'Weeks', 'Age', 'Sex', 'expected_fvc', 'SmokingStatus']]

patient_df['num_weeks'] = train[['Patient', 'Weeks']].groupby('Patient').transform('count')

patient_df['min_week'] = train[['Patient', 'Weeks']].groupby('Patient').transform('min')

_le = LabelEncoder()

patient_df['SmokingStatus'] = _le.fit_transform(train['SmokingStatus'])

patient_df['Sex'] = (patient_df['Sex'] == 'Male').astype(int)



patient_df['Age'] = replace_with_frequency_bins(patient_df['Age'], 5)

patient_df['num_weeks'] = replace_with_frequency_bins(patient_df['num_weeks'], 5)

patient_df['min_week'] = replace_with_frequency_bins(patient_df['min_week'], 5)

patient_df['expected_fvc'] = pd.Series(replace_with_frequency_bins(patient_df['expected_fvc'], 6)).astype(str)



patient_df = patient_df.drop('Weeks',axis=1).drop_duplicates()

patient_df.index = list(range(len(patient_df)))
patient_df = patient_df.merge(image_feature_df, left_on='Patient', right_on='patient_id').drop('patient_id', axis=1)
class LungDataset(Dataset):

    def __init__(self, mode, data, image_features, preprocessing_params=None, expand=True):

        data = data.drop_duplicates(subset=['Patient', 'Weeks'])

        self.inference_data = None

        self.ohe = False

        self.preprocessing_params = preprocessing_params if preprocessing_params is not None else {

            'min_week': -12,

            'max_week': 133

        }

        self.expand = expand

        self.raw_data = self.process_data(data.copy(), image_features, mode, self.preprocessing_params)

        self.data = self.raw_data[self.features]

        self.target = self.raw_data['FVC'] if mode != 'test' else None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.inference = False

        self.pass_y = True if mode != 'test' else False

        

    def expand_train_data(self, train):

        train_expanded = pd.DataFrame()

        for patient, patient_df in train.groupby('Patient'):

            patient_expanded = pd.DataFrame()

            expected_fvc = np.mean((100 * patient_df['FVC']) / patient_df['Percent'])

            for week in patient_df['Weeks'].sort_values():

                week_df = patient_df.copy()

                week_df['weeks_from_first_visit'] = week_df['Weeks'] - week

                week_df['first_test_fvc'] = patient_df.loc[patient_df['Weeks'] == week, 'FVC'].mean()

                week_df['first_test_week'] = week

                week_df['predict_week'] = patient_df['Weeks']

                week_df.drop(['Percent', 'Weeks'], axis=1, inplace=True)

                patient_expanded = pd.concat([patient_expanded, week_df], ignore_index=True)

            patient_expanded['expected_fvc'] = expected_fvc

            train_expanded = pd.concat([train_expanded, patient_expanded], ignore_index=True)

            train_expanded = train_expanded[(train_expanded['weeks_from_first_visit'] >= -12) & (train_expanded['weeks_from_first_visit'] <= 133)]

        return train_expanded



    def process_data(self, data, img_data, mode, params):

        if mode == 'train' and self.expand:

            data = self.expand_train_data(data)

        else:

            # expected_FVC

            data['expected_fvc'] = (100 * data['FVC']) / data['Percent']

            data['first_test_week'] = data[['Patient', 'Weeks']].groupby('Patient').transform('min')

            data['weeks_from_first_visit'] = data['Weeks'] - data['first_test_week']

            data['predict_week'] = data['Weeks']

            min_fvc_df = data.sort_values('first_test_week').groupby('Patient').head(1)[['Patient', 'FVC']]

            min_fvc_df.columns = ['Patient', 'first_test_fvc']

            data = data.merge(min_fvc_df, on='Patient', how='left')

        data['percent'] = data['first_test_fvc'] / data['expected_fvc']

            

        

        # age

        min_age = params.get('min_age', data['Age'].min())

        max_age = params.get('max_age', data['Age'].max())

        data['age'] = (data['Age'] - data['Age'].min()) / (data['Age'].max() - data['Age'].min())

        

        # OHE

        if self.ohe:

            sex_df = pd.get_dummies(data['Sex']).reset_index(drop=True)

            sex_columns = ['Male', 'Female']

            for col in sex_columns:

                if col not in sex_df:

                    sex_df[col] = 0

            smoke_df = pd.get_dummies(data['SmokingStatus']).reset_index(drop=True)

            smoke_columns = ['Currently smokes','Ex-smoker','Never smoked']

            for col in smoke_columns:

                if col not in smoke_df.columns:

                    smoke_df[col] = 0

                    

        else:

            sex_map = {'Male': 0, 'Female': 1}

            smoke_map = {'Ex-smoker': 0, 'Never smoked': 1, 'Currently smokes': 2}

            data['sex'] = data['Sex'].map(sex_map)

            data['smoke'] = data['SmokingStatus'].map(smoke_map)

            

        # base week

        min_week = params.get('min_week')

        max_week = params.get('max_week')

        data['first_test_week'] = (data['first_test_week'] - min_week) / (max_week - min_week)

        data['weeks_from_first_visit'] = (data['weeks_from_first_visit'] - min_week) / (max_week - min_week)

        data['predict_week'] = (data['predict_week'] - min_week) / (max_week - min_week)

        

        # Opting min_FVC value

        min_FVC = params.get('min_fvc', min(data['expected_fvc'].min(), data['FVC'].min()))

        max_FVC = params.get('max_fvc', max(data['expected_fvc'].max(), data['FVC'].max()))

        data['expected_fvc'] = (data['expected_fvc'] - min_FVC) / (max_FVC - min_FVC)

        data['first_test_fvc'] = (data['first_test_fvc'] - min_FVC) /(max_FVC - min_FVC)

        data['FVC'] = (data['FVC'] - min_FVC) /(max_FVC - min_FVC)

    

        #print(data.shape, sex_df.shape, smoke_df.shape)

        data.reset_index(drop=True, inplace=True)

        if self.ohe:

            data = pd.concat([data, sex_df, smoke_df], axis=1)

        #print(data.shape)

        # update params

        self.preprocessing_params = {

            'min_age': min_age,

            'max_age': max_age,

            'min_week': min_week,

            'max_week': max_week,

            'min_fvc': min_FVC,

            'max_fvc': max_FVC

        }

        

        self.features = ['first_test_fvc',

                         'age', 'predict_week', 'percent', 'expected_fvc', 'weeks_from_first_visit'] #'first_test_week']

        

        #Image features

        if img_data is not None:

            # lung_volume

            missing_img_data = img_data[img_data['missing'] == 1].copy()

            img_data = img_data[img_data['missing'] == 0].copy()

            min_lung_vol = params.get('min_lung_vol', img_data['lung_volume_in_liters'].min())

            max_lung_vol = params.get('max_lung_vol', img_data['lung_volume_in_liters'].max())

            img_data['lung_volume'] = (img_data['lung_volume_in_liters'] - min_lung_vol) / (max_lung_vol - min_lung_vol)

            

            # lung height

            min_lung_height = params.get('min_lung_height', img_data['height_of_lung_cm'].min())

            max_lung_height = params.get('max_lung_height', img_data['height_of_lung_cm'].max())

            img_data['lung_height'] = (img_data['height_of_lung_cm'] - min_lung_height) / (max_lung_height - min_lung_height)

            

            #mean HU value

            min_hu_mean = params.get('min_hu_mean', img_data['lung_mean_hu'].min())

            max_hu_mean = params.get('max_hu_mean', img_data['lung_mean_hu'].max())

            img_data['hu_mean'] = (img_data['lung_mean_hu'] - min_hu_mean) / (max_hu_mean - min_hu_mean)

            

            #skew

            min_hu_skew = params.get('min_hu_skew', img_data['lung_skew'].min())

            max_hu_skew = params.get('max_hu_skew', img_data['lung_skew'].max())

            img_data['hu_skew'] = (img_data['lung_skew'] - min_hu_skew) / (max_hu_skew - min_hu_skew)

        

            #kurtosis

            min_hu_kurtosis = params.get('min_hu_kurtosis', img_data['lung_kurtosis'].min())

            max_hu_kurtosis = params.get('max_hu_kurtosis', img_data['lung_kurtosis'].max())

            img_data['hu_kurtosis'] = (img_data['lung_kurtosis'] - min_hu_kurtosis) / (max_hu_kurtosis - min_hu_kurtosis)

            

            img_features = ['lung_volume', 'lung_height', 'hu_mean', 'hu_skew', 'hu_kurtosis', 'missing']

            self.features += img_features

            self.preprocessing_params.update({

                'min_lung_vol': min_lung_vol,

                'max_lung_vol': max_lung_vol,

                'min_lung_height': min_lung_height,

                'max_lung_height': max_lung_height,

                'min_hu_mean': min_hu_mean,

                'max_hu_mean': max_hu_mean,

                'min_hu_skew': min_hu_skew,

                'max_hu_skew': max_hu_skew,

                'min_hu_kurtosis': min_hu_kurtosis,

                'max_hu_kurtosis': max_hu_kurtosis,

            })

            bin_features = [x for x in img_data.columns if 'bin_' in x]

            self.features += bin_features

            img_feature_df = img_data[[*img_features, *bin_features, 'patient_id']]

            for col in img_feature_df.columns:

                if col not in missing_img_data:

                    missing_img_data[col] = 0

            img_feature_df = pd.concat([img_feature_df, missing_img_data], ignore_index=True)

            img_feature_df = img_feature_df.fillna(0)

            data = data.merge(img_feature_df, left_on='Patient', right_on='patient_id')

        

        if self.ohe:

            self.features += [*sex_df.columns, *smoke_df.columns]

        else:

            self.features += ['sex', 'smoke']

        data.index = list(range(len(data)))

        

        # construct inference data

        if mode != 'train':

            inference_df = pd.DataFrame(

                itertools.product(data['Patient'].unique(), range(-12, 134)), columns=['Patient', 'Weeks']

            )

            min_week_df = data[data['weeks_from_first_visit'] == 12 / (133 + 12)].drop('Weeks', axis=1)

            inference_df = inference_df.merge(min_week_df, on='Patient', how='left')

            first_test_week = (inference_df['first_test_week'] * (max_week - min_week)) + min_week

            inference_df['weeks_from_first_visit'] = inference_df['Weeks'] - first_test_week

            inference_df['predict_week'] = inference_df['Weeks']

            inference_df['weeks_from_first_visit'] = (inference_df['weeks_from_first_visit'] - min_week) / (max_week - min_week)

            inference_df['predict_week'] = (inference_df['predict_week'] - min_week) / (max_week - min_week)

            self.inference_data = inference_df.reset_index(drop=True)

            

        return data

    

    def get_params(self):

        return self.preprocessing_params

    

    def get_inference_df(self, last_3_visits=False):

        return self.inference_data

    

    def set_inference(self, value):

        self.inference = value

        

    def get_y(self, value):

        self.pass_y = value

    

    def __len__(self):

        if not self.inference:

            return len(self.data)

        return len(self.inference_data)

    

    def __getitem__(self, idx):

        if not self.inference:

            data = self.data

        else:

            data = self.inference_data[self.features]

        if (self.target is None) or self.inference or (not self.pass_y):

            return torch.tensor(data.loc[idx].values, dtype=torch.float32).to(self.device)

        return torch.tensor(data.loc[idx].values, dtype=torch.float32).to(self.device), torch.tensor(self.target[idx], dtype=torch.float32).to(self.device)
train_dataset = LungDataset('train', train, image_feature_df, expand=False)
train_dataset.get_params()
def get_splits(stratify_df, train_df, image_df, kfolds, stratify_col=None, expand=True):

    splits = kfolds.split(stratify_df) if not stratify_col else kfolds.split(stratify_df, stratify_df[stratify_col])

    for train_index, valid_index in splits:

        train_patients = patient_df.loc[train_index, 'Patient']

        valid_patients = patient_df.loc[valid_index, 'Patient']

        batch_train_df = train_df[train_df['Patient'].isin(train_patients)]

        batch_valid_df = train_df[train_df['Patient'].isin(valid_patients)]

        batch_train_image_df = image_df[image_df['patient_id'].isin(train_patients)]

        batch_valid_image_df = image_df[image_df['patient_id'].isin(valid_patients)]

        train_dataset = LungDataset('train', batch_train_df, batch_train_image_df, expand=expand)

        params = train_dataset.get_params()

        valid_dataset = LungDataset('valid', batch_valid_df, batch_valid_image_df, params)

        yield train_dataset, valid_dataset
from pytorch_lightning.callbacks import Callback





class CVResultTracker(Callback):

    def __init__(self, result_file, cv_result_file):

        super().__init__()

        self.result_file = result_file

        self_cv_result_file = cv_result_file

        

    def write_file(self, path, df):

        if path.is_file():

            existing_df = pd.read_csv(path)

            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(path, index=False)

        return df

        

    def teardown(self, trainer, pl_module, stage):

        logger = None

        for logr in trainer.logger:

            logger = logr if 'CSVLogger' in str(logr.__class__) else None

        if logger:

            metrics_df = pd.read_csv(f'./{logger.save_dir}/{logger.name}/{logger.version}/metrics.csv')

            val_df = metrics_df.dropna(subset=['val_loss'])

            val_df['name'] = logger.name + '_' + logger.version

            val_df['model_name'] = logger.name.split('fold')[0]

            val_df_last = val_df[val_df['epoch'] == val_df['epoch'].max()]

            val_df_best = val_df[val_df['val_nll'] == val_df['val_nll'].min()]

            val_df_last = val_df_last.dropna(axis=1, how='all')

            val_df_best = val_df_best.dropna(axis=1, how='all')

            fold_results_best_path = Path(f'./{logger.save_dir}/fold_results_best.csv')

            fold_results_last_path = Path(f'./{logger.save_dir}/fold_results_last.csv')

            model_results_best_path = Path(f'./{logger.save_dir}/model_results_best.csv')

            model_results_last_path = Path(f'./{logger.save_dir}/model_results_last.csv')

            fold_results_best_df = self.write_file(fold_results_best_path, val_df_best)

            fold_results_last_df = self.write_file(fold_results_last_path, val_df_last)

            

            model_best_df = fold_results_best_df.drop('name', axis=1)

            model_last_df = fold_results_last_df.drop('name', axis=1)

            

            model_best_df = model_best_df.groupby('model_name', as_index=False).mean()

            model_last_df = model_last_df.groupby('model_name', as_index=False).mean()

            

            model_best_df.to_csv(model_results_best_path, index=False)

            model_last_df.to_csv(model_results_last_path, index=False)

            

            
import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F



import pytorch_lightning as pl

from pytorch_lightning.metrics.metric import TensorMetric

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.callbacks import LearningRateLogger





from pytorch_lightning import loggers







class NLL(TensorMetric):

    name = 'NLL'

    def __init__(self, min_fvc, max_fvc):

        super().__init__(name='nll')

        self.min_fvc = min_fvc

        self.max_fvc = max_fvc

        

    def forward(self, preds, target):

        preds = (preds * (self.max_fvc - self.min_fvc)) + self.min_fvc

        target = (target * (self.max_fvc - self.min_fvc)) + self.min_fvc

        sigma = preds[:, 2] - preds[:, 0]

        fvc_pred = preds[:, 1]



        #sigma_clip = sigma + C1

        sigma_clip = torch.max(sigma, torch.tensor(70.).to(self.device))

        delta = torch.abs(target - fvc_pred)

        delta = torch.min(delta, torch.tensor(1000.).to(self.device))

        sq2 = torch.sqrt(torch.tensor(2.).to(device))

        metric = (delta / sigma_clip)*sq2 + torch.log(sigma_clip* sq2)

        # print(torch.mean(metric))

        return torch.mean(metric)

    

class MAE(TensorMetric):

    name = 'MAE'

    def __init__(self, min_fvc, max_fvc):

        super().__init__(name='MAE')

        self.min_fvc = min_fvc

        self.max_fvc = max_fvc

        

    def forward(self, preds, target):

        preds = (preds * (self.max_fvc - self.min_fvc)) + self.min_fvc

        target = (target * (self.max_fvc - self.min_fvc)) + self.min_fvc

        return torch.mean(torch.abs(target - preds[:, 1]))

    



class LQuantModel(pl.LightningModule):

    def __init__(self, learning_rate, total_steps, train_params, in_tabular_features=8,

                 out_quantiles=3):

        super(LQuantModel, self).__init__()

        self.learning_rate = learning_rate

        self.total_steps = total_steps

        self.fc1 = nn.Linear(in_tabular_features, 512)

        self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, out_quantiles)

        self.fc4 = nn.Linear(256, out_quantiles)

        self.train_params = train_params

        self.save_hyperparameters()

        self.metrics = [

            NLL(

                train_params.get('min_fvc'),

                train_params.get('max_fvc'),

            ),

            MAE(

                train_params.get('min_fvc'),

                train_params.get('max_fvc'),

            )

        ]

        self.eval_tracker = {}

    

        

    def forward(self, x):

        x = F.relu(self.fc1(x))   

        x = F.relu(self.fc2(x))

        x1 = self.fc3(x)

        x2 = F.relu(self.fc4(x))

        preds = x1 + torch.cumsum(x2, axis=1)

        return preds

    

    def configure_optimizers(self):

        optimizer = optim.SGD(self.parameters(), lr=(self.learning_rate), weight_decay=0.01)

        scheduler = optim.lr_scheduler.OneCycleLR(

            optimizer,

            max_lr=self.learning_rate,

            total_steps=self.total_steps

        )

        return [optimizer], [{

            'scheduler': scheduler,

            'interval': 'step',

            'frequency': 1

        }]

        

    def set_loss_fn(self, loss_fn):

        self.loss_fn = loss_fn

        

        

    def training_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        result = pl.TrainResult(minimize=loss)

        result.log('train_loss', loss, prog_bar=True)

        for metric in self.metrics:

            metric.to(self.device)

            metric_value = metric(y_hat, y)

            result.log(f'train_{metric.name}', metric_value, on_epoch=True, on_step=False)

        return result



    def validation_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        result = pl.EvalResult(checkpoint_on=loss)

        result.log('val_loss', loss, on_epoch=True, on_step=False)

        for metric in self.metrics:

            metric.to(self.device)

            metric_value = metric(y_hat, y)

            result.log(f'val_{metric.name}', metric_value, on_epoch=True, on_step=False)

        return result

    

#     def validation_epoch_end(self, validation_results):

#         self.eval_tracker = {k: v[-1] for k, v in validation_results.items() if k[0:4] == 'val_'}

#         return validation_results

    

    def test_step(self, batch, batch_idx):

        y_hat = self(batch)

        result = pl.EvalResult()

        result.predictions = y_hat

        return result

    

    

    

class LTabnet(pl.LightningModule):

    def __init__(self, learning_rate, total_steps, train_params, in_tabular_features=8,

                 out_quantiles=3):

        super(LTabnet, self).__init__()

        self.learning_rate = learning_rate

        self.total_steps = total_steps

        self.tabnet = TabNet(

            input_dim=26,

            output_dim=3,

            n_a=64,

            n_d=64,

            cat_idxs=[24,25],

            cat_dims=[2,3],

            cat_emb_dim=[4,4],

        ).to(self.device)

        self.train_params = train_params

        self.save_hyperparameters()

        self.metrics = [

            NLL(

                train_params.get('min_fvc'),

                train_params.get('max_fvc'),

            ),

            MAE(

                train_params.get('min_fvc'),

                train_params.get('max_fvc'),

            ),

        ]

    

        

    def forward(self, x):

        preds, _ = self.tabnet(x)

        return preds

    

    def configure_optimizers(self):

        optimizer = optim.SGD(self.parameters(), lr=(self.learning_rate), weight_decay=0.01)

        scheduler = optim.lr_scheduler.OneCycleLR(

            optimizer,

            max_lr=self.learning_rate,

            total_steps=self.total_steps

        )

        return [optimizer], [{

            'scheduler': scheduler,

            'interval': 'step',

            'frequency': 1

        }]

        

    def set_loss_fn(self, loss_fn):

        self.loss_fn = loss_fn

        

        

    def training_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        result = pl.TrainResult(minimize=loss)

        result.log('train_loss', loss, prog_bar=True)

        for metric in self.metrics:

            metric.to(self.device)

            metric_value = metric(y_hat, y)

            result.log(f'train_{metric.name}', metric_value, on_epoch=True, on_step=False)

        return result



    def validation_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        result = pl.EvalResult(checkpoint_on=loss)

        result.log('val_loss', loss, on_epoch=True, on_step=False)

        for metric in self.metrics:

            metric.to(self.device)

            metric_value = metric(y_hat, y)

            result.log(f'val_{metric.name}', metric_value, on_epoch=True, on_step=False)

        return result

    

    def test_step(self, batch, batch_idx):

        y_hat = self(batch)

        result = pl.EvalResult()

        result.predictions = y_hat

        return result
class LungDataModule(pl.LightningDataModule):

    def __init__(self, train_dataset, valid_dataset, test_dataset, batch_size=128):

        super().__init__()

        self.train_dataset = train_dataset

        self.valid_dataset = valid_dataset

        self.test_dataset = test_dataset

        self.batch_size = batch_size

        

    def train_dataloader(self):

        return DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)

    

    def val_dataloader(self):

        return DataLoader(valid_dataset, shuffle=False, batch_size=self.batch_size)

    

    def test_dataloader(self):

        return DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size)

    

    



def predict_dataloader(model, dataloader):

    predictions = []

    model.eval()

    for batch in dataloader:

        predictions.append(model(batch).detach().cpu().numpy())

    return np.vstack(predictions)
def quantile_loss(preds, target, quantiles, device):

    #assert not target.requires_grad

    #assert preds.size(0) == target.size(0)

    losses = []

    q = torch.tensor(quantiles).to(device).repeat(preds.shape[0], 1)

    errors = target.unsqueeze(1).repeat(1, 3) - preds

    losses = torch.max((q - 1) * errors, q * errors)

    return torch.mean(losses).unsqueeze(0)





def metric_score(preds, target, max_fvc, min_fvc, device):

    preds = (preds * (max_fvc - min_fvc)) + min_fvc

    target = (target * (max_fvc - min_fvc)) + min_fvc

    sigma = preds[:, 2] - preds[:, 0]

    fvc_pred = preds[:, 1]

    

    #sigma_clip = sigma + C1

    sigma_clip = torch.max(sigma, torch.tensor(70.).to(device))

    delta = torch.abs(target - fvc_pred)

    delta = torch.min(delta, torch.tensor(1000.).to(device))

    sq2 = torch.sqrt(torch.tensor(2.).to(device))

    metric = (delta / sigma_clip)*sq2 + torch.log(sigma_clip* sq2)

    # print(torch.mean(metric))

    return torch.mean(metric)



def metric_loss(preds, target, _lambda, quantiles, max_fvc, min_fvc, device):

    return ((_lambda * quantile_loss(preds, target, quantiles, device)) + 

            ((1 - _lambda) * metric_score(preds, target, max_fvc, min_fvc, device)))



seed_everything(2020)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

_skf = StratifiedKFold(n_splits=5)

_kf = KFold()

lr_monitor_callback = LearningRateLogger(logging_interval='step')
seed_everything(2020)



name = 'tabnet_img_featuresall_skf_mloss1_lr5e-3_faster_missing'

trained_models = []

cv_predictions = []

epochs = 350

batch_size = 128





for i, (train_dataset, valid_dataset) in enumerate(get_splits(patient_df, train, image_feature_df, _skf, 'expected_fvc',expand=True)):

    total_steps = epochs * math.ceil((len(train_dataset) / batch_size))

    train_params = train_dataset.get_params()

    loss_fn = partial(

        metric_loss,

        _lambda=1,

        quantiles=(0.3, 0.5, 0.7),

        max_fvc=train_params['max_fvc'],

        min_fvc=train_params['min_fvc'],

        device=device

    )

    pathdir = Path(f'./models/{name}/fold_{i}/')

    pathdir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(

        filepath=pathdir,

        save_top_k=1,

        save_last=True,

        verbose=False,

        monitor='val_nll',

        mode='min',

        prefix=''

    )



    cv_result_tracker = CVResultTracker('./fold_results.csv', './model_results.csv')

    model = LTabnet(5e-3, total_steps, train_params, in_tabular_features=29).to(device)

    model.set_loss_fn(loss_fn)

    test_dataset = LungDataset('test', test, None, train_params)

    test_dataset.set_inference(True)

    data_module = LungDataModule(train_dataset, valid_dataset, test_dataset, batch_size=batch_size)

    

    tb_logger = loggers.TensorBoardLogger('lightning_logs/', name=f'{name}', version=f'fold_{i}')

    csv_logger = loggers.CSVLogger('results', name=name, version=f'fold_{i}')

    

    trainer = pl.Trainer(

        max_epochs=epochs, 

        check_val_every_n_epoch=5, 

        logger=[tb_logger, csv_logger],

        checkpoint_callback=checkpoint_callback,

        gpus=1,

        callbacks=[lr_monitor_callback, cv_result_tracker]

    )

    trainer.fit(model, data_module)

    trained_models.append(model)
!ls ./models
!ls ./results