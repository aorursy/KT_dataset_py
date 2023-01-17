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

        return h['image'][:][:, :, :, 0] # we read the data from the file
sample_scan = train_df.iloc[0] # just take the first row

print(sample_scan)

# turn the h5_path into the full path

image_data = read_scan(sample_scan['h5_path'])

print('Image Shape:', image_data.shape)
plt.hist(image_data[image_data>0], 100);
def calc_threshold_volume(in_image_data):

    return np.sum(in_image_data>600)

print(calc_threshold_volume(image_data))
%%time

train_df['white_matter_volume'] = train_df['h5_path'].map(lambda c_filename: calc_threshold_volume(read_scan(c_filename)))
sns.lmplot('age_years', 'white_matter_volume', data=train_df)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(train_df['white_matter_volume'].values.reshape((-1, 1)), train_df['age_years'].values)

lin_reg.coef_
test_df = pd.read_csv('../input/test_sample_submission.csv')[['scan_id']]

test_df['h5_path'] = test_df['scan_id'].map(lambda s_id: 'mri_{:08d}.h5'.format(s_id))

test_df['white_matter_volume'] = test_df['h5_path'].map(lambda c_filename: calc_threshold_volume(read_scan(c_filename, folder='test')))

test_df.head(5)
test_df['age_years'] = lin_reg.predict(test_df['white_matter_volume'].values.reshape((-1, 1)))

sns.lmplot('age_years', 'white_matter_volume', data=test_df)
test_df['age_years'].describe()
# save the output

test_df[['scan_id', 'age_years']].to_csv('threshold_prediction.csv', index=False)