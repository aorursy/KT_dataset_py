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
plt.hist(image_data.ravel(), 30)
noisy_data = image_data+np.random.uniform(-1, 1, size=image_data.shape)

bins = [np.percentile(noisy_data, k) for k in np.linspace(70, 99, 25)]

print(bins)

def calc_brightness_histogram(in_image_data):

    return np.histogram(in_image_data.ravel(), bins)[0]

plt.bar(bins[1:], calc_brightness_histogram(image_data), width=50)

calc_brightness_histogram(image_data)
%%time

train_df['brightness'] = train_df['h5_path'].map(lambda c_filename: calc_brightness_histogram(read_scan(c_filename)))
train_df.sample(3)
plt.imshow(np.stack(train_df['brightness'].values, 0))
from sklearn.ensemble import RandomForestRegressor

age_model = RandomForestRegressor()

age_model.fit(np.stack(train_df['brightness'].values, 0), train_df['age_years'].values)

age_model
plt.plot(

    train_df['age_years'].values,

    age_model.predict(np.stack(train_df['brightness'].values, 0)),

    '.'

)

plt.xlabel('Actual Age (Years)')

plt.ylabel('Predicted Age (Years)')
test_df = pd.read_csv('../input/test_sample_submission.csv')[['scan_id']]

test_df['h5_path'] = test_df['scan_id'].map(lambda s_id: 'mri_{:08d}.h5'.format(s_id))

test_df['brightness'] = test_df['h5_path'].map(

    lambda c_filename: calc_brightness_histogram(read_scan(c_filename, folder='test'))

)

test_df.head(5)
test_df['age_years'] = age_model.predict(np.stack(test_df['brightness'].values, 0))

plt.hist(test_df['age_years'])
# save the output

test_df[['scan_id', 'age_years']].to_csv('histogram_brightness_prediction.csv', index=False)