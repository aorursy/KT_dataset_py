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
from keras import layers, models, optimizers, losses

simple_model = models.Sequential(name='ShallowCNN')

simple_model.add(layers.BatchNormalization(input_shape=(image_data.shape)+(1,)))

# block 1

simple_model.add(layers.Conv3D(8, (3, 3, 3), activation='linear'))

simple_model.add(layers.BatchNormalization())

simple_model.add(layers.Activation('relu'))

simple_model.add(layers.MaxPool3D((2, 2, 2)))

# block 2

simple_model.add(layers.Conv3D(16, (3, 3, 3), activation='linear'))

simple_model.add(layers.BatchNormalization())

simple_model.add(layers.Activation('relu'))

simple_model.add(layers.MaxPool3D((2, 2, 2)))

# block 3

simple_model.add(layers.Conv3D(32, (3, 3, 3), activation='linear'))

simple_model.add(layers.BatchNormalization())

simple_model.add(layers.Activation('relu'))

simple_model.add(layers.MaxPool3D((2, 2, 2)))

# block 4

simple_model.add(layers.Conv3D(64, (3, 3, 3), activation='linear'))

simple_model.add(layers.BatchNormalization())

simple_model.add(layers.Activation('relu'))

simple_model.add(layers.MaxPool3D((2, 2, 2)))

# block 4

simple_model.add(layers.Conv3D(128, (3, 3, 3), activation='linear'))

simple_model.add(layers.BatchNormalization())

simple_model.add(layers.Activation('relu'))

simple_model.add(layers.MaxPool3D((2, 2, 2)))

# add elements together

simple_model.add(layers.GlobalAvgPool3D())

simple_model.add(layers.Dropout(0.5))

simple_model.add(layers.Dense(256, activation='tanh'))

simple_model.add(layers.Dense(1, activation='linear'))



# setup model

simple_model.compile(optimizer=optimizers.Adam(1e-3), 

                     loss=losses.mean_squared_error, 

                     metrics=['mae'])



simple_model.summary()
from keras.utils import vis_utils

from IPython.display import SVG

SVG(vis_utils.model_to_dot(simple_model, show_shapes=True).create_svg())
def data_gen_func(in_df, batch_size=8):

    """Generate image and age label data in batches"""

    while True:

        image, age = [], []

        balanced_sample_df = in_df.groupby(in_df['age_years']<40).apply(lambda x: x.sample(batch_size//2)).reset_index(drop=True)

    

        for _, c_row in balanced_sample_df.iterrows():

            age += [c_row['age_years']]

            image += [read_scan(c_row['h5_path'])]

        yield np.expand_dims(np.stack(image, 0), -1), np.expand_dims(np.stack(age), -1)

train_gen = data_gen_func(train_df)

X, y = next(train_gen)

print(X.shape, y.shape)
train_df['prediction_0'] = train_df['h5_path'].map(lambda c_filename: 

                                               simple_model.predict(

                                                   np.expand_dims(

                                                       np.expand_dims(read_scan(c_filename), -1),

                                                       0

                                                   )

                                               )[0,0]

                                              )

sns.lmplot('age_years', 'prediction_0', data=train_df)
for epoch in range(1, 20):

    # train model

    simple_model.fit_generator(train_gen, 

                               steps_per_epoch=12, 

                               verbose=True, 

                               epochs=1)

    # keep track of results

    train_df['prediction_{}'.format(epoch)] = train_df['h5_path'].map(lambda c_filename: 

                                                   simple_model.predict(

                                                       np.expand_dims(

                                                           np.expand_dims(read_scan(c_filename), -1),

                                                           0

                                                       )

                                                   )[0,0]

                                                  )
sns.lmplot('age_years', 'prediction_19', data=train_df)
flat_train_df = pd.melt(train_df, id_vars=['scan_id', 'age_years', 'ni_path', 'h5_path'])

flat_train_df['epoch'] = flat_train_df['variable'].map(lambda x: int(x.split('_')[-1]))

lm = sns.lmplot(

    'age_years','value', 

    hue='epoch',

    data=flat_train_df

)

lm.axes[0,0].set_xlim(0, 100)

lm.axes[0,0].set_ylim(0, 100)
test_df = pd.read_csv('../input/test_sample_submission.csv')[['scan_id']]

test_df['h5_path'] = test_df['scan_id'].map(lambda s_id: 'mri_{:08d}.h5'.format(s_id))

test_df['age_years'] = test_df['h5_path'].map(lambda c_filename: 

                                               simple_model.predict(

                                                   np.expand_dims(

                                                       np.expand_dims(read_scan(c_filename, folder='test'), -1),

                                                       0

                                                   )

                                               )[0,0]

                                              )

test_df.head(5)
test_df['age_years'].hist()
# save the output

test_df[['scan_id', 'age_years']].to_csv('cnn3d_prediction.csv', index=False)