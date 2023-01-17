!pip install pyradiomics
# special functions for using pyradiomics

from SimpleITK import GetImageFromArray

import radiomics

from radiomics.featureextractor import RadiomicsFeatureExtractor # This module is used for interaction with pyradiomic

import logging

logging.getLogger('radiomics').setLevel(logging.CRITICAL + 1)  # this tool makes a whole TON of log noise
# Instantiate the extractor

texture_extractor = RadiomicsFeatureExtractor(verbose=False)

texture_extractor.disableAllFeatures()

_text_feat = {ckey: [] for ckey in texture_extractor.featureClassNames}

texture_extractor.enableFeaturesByName(**_text_feat)



print('Extraction parameters:\n\t', texture_extractor.settings)

print('Enabled filters:\n\t', texture_extractor.enabledImagetypes) 

print('Enabled features:\n\t', texture_extractor.enabledFeatures) 
import numpy as np # for manipulating 3d images

import pandas as pd # for reading and writing tables

import h5py # for reading the image files

import skimage # for image processing and visualizations

import sklearn # for machine learning and statistical models

import os # help us load files and deal with paths
%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

plt.rcParams["figure.figsize"] = (8, 8)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})
train_df = pd.read_csv('../input/train.csv')

train_df.head(5) # show the first 5 lines
def read_scan(in_filename, folder='train'):

    full_scan_path = os.path.join('..', 'input',folder, in_filename)

    # load the image using hdf5

    with h5py.File(full_scan_path, 'r') as h:

        # [::2, ::4, ::4, 0] downsampling makes it go much faster

        return h['image'][:][:, :, :, 0] # we read the data from the file
sample_scan = train_df.iloc[0] # just take the first row

print(sample_scan)

# turn the h5_path into the full path

image_data = read_scan(sample_scan['h5_path'])

print('Image Shape:', image_data.shape)
# we take a mask by just keeping the part of the image greater than 0

plt.imshow(np.sum((image_data>0).astype(float), 0))
%%time

results = texture_extractor.execute(GetImageFromArray(image_data),

                            GetImageFromArray((image_data>0).astype(np.uint8)))
pd.DataFrame([results]).T
def calc_radiomics(in_image_data):

    return texture_extractor.execute(GetImageFromArray(in_image_data),

                            GetImageFromArray((in_image_data>0).astype(np.uint8)))
%%time

train_df['radiomics'] = train_df['h5_path'].map(lambda c_filename: calc_radiomics(read_scan(c_filename)))
new_train_df = pd.DataFrame([dict(**c_row.pop('radiomics'), **c_row) for _, c_row in train_df.iterrows()])

print(new_train_df.shape, 'data prepared')

new_train_df.sample(3)
# leave out anything that doesn't start with original (just junk from the input)

# also remove shape since it is not very informative

value_feature_names = [c_col for c_col in full_df.columns if (c_col.startswith('original') and '_shape_' not in c_col)]

print(np.random.choice(value_feature_names, 3), 'of', len(value_feature_names))
from sklearn.ensemble import RandomForestRegressor

age_model = RandomForestRegressor()

age_model.fit(np.stack(new_train_df[value_feature_names].values, 0), 

              new_train_df['age_years'].values)

age_model
plt.plot(

    new_train_df['age_years'].values,

    age_model.predict(np.stack(new_train_df[value_feature_names].values, 0)),

    '.'

)

plt.xlabel('Actual Age (Years)')

plt.ylabel('Predicted Age (Years)')
test_df = pd.read_csv('../input/test_sample_submission.csv')[['scan_id']]

test_df['h5_path'] = test_df['scan_id'].map(lambda s_id: 'mri_{:08d}.h5'.format(s_id))

test_df['radiomics'] = test_df['h5_path'].map(

    lambda c_filename: calc_radiomics(read_scan(c_filename, folder='test'))

)

new_test_df = pd.DataFrame([dict(**c_row.pop('radiomics'), **c_row) for _, c_row in test_df.iterrows()])

new_test_df.sample(3)
new_test_df['age_years'] = age_model.predict(np.stack(new_test_df[value_feature_names].values, 0))

plt.hist(new_test_df['age_years'])
# save the output

new_test_df[['scan_id', 'age_years']].to_csv('radiomics_prediction.csv', index=False)