!mkdir ~/.keras

!mkdir ~/.keras/models

!cp ../input/keras-pretrained-models/*notop* ~/.keras/models/

!cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/

!cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # showing and rendering figures

# io related

from skimage.io import imread

import os

from glob import glob

# not needed in Kaggle, but required in Jupyter

%matplotlib inline 
base_bone_dir = os.path.join('..', 'input', 'rsna-bone-age')

age_df = pd.read_csv(os.path.join(base_bone_dir, 'boneage-training-dataset.csv'))

age_df['path'] = age_df['id'].map(lambda x: os.path.join(base_bone_dir,

                                                         'boneage-training-dataset', 

                                                         'boneage-training-dataset', 

                                                         '{}.png'.format(x)))

age_df['exists'] = age_df['path'].map(os.path.exists)

print(age_df['exists'].sum(), 'images found of', age_df.shape[0], 'total')

age_df['gender'] = age_df['male'].map(lambda x: 'male' if x else 'female')

boneage_mean = age_df['boneage'].mean()

boneage_div = 2*age_df['boneage'].std()

age_df['boneage_zscore'] = age_df['boneage'].map(lambda x: (x-boneage_mean)/boneage_div)

age_df.dropna(inplace = True)

age_df.drop(['exists'],1, inplace = True)

age_df.sample(3)
age_df[['boneage', 'male', 'boneage_zscore']].hist(figsize = (10, 5))

age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)
from sklearn.model_selection import train_test_split

raw_train_df, valid_df = train_test_split(age_df, 

                                   test_size = 0.25, 

                                   random_state = 2018,

                                   stratify = age_df['boneage_category'])

print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
train_df = raw_train_df.groupby(['boneage_category', 'male']).apply(lambda x: x.sample(500, replace = True)

                                                      ).reset_index(drop = True)

print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])

train_df[['boneage', 'male']].hist(figsize = (10, 5))
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.imagenet_utils import preprocess_input

IMG_SIZE = (256, 256) # reasonable resolution images

core_idg = ImageDataGenerator(samplewise_center=False, 

                              samplewise_std_normalization=False, 

                              horizontal_flip = True, 

                              vertical_flip = False, 

                              height_shift_range = 0.15, 

                              width_shift_range = 0.15, 

                              rotation_range = 5, 

                              shear_range = 0.01,

                              fill_mode = 'reflect',

                              zoom_range=0.25,

                             preprocessing_function = preprocess_input)
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):

    base_dir = os.path.dirname(in_df[path_col].values[0])

    print('## Ignore next message from keras, values are replaced anyways')

    df_gen = img_data_gen.flow_from_directory(base_dir, 

                                     class_mode = 'sparse',

                                    **dflow_args)

    df_gen.filenames = in_df[path_col].values

    df_gen.classes = np.stack(in_df[y_col].values)

    df_gen.samples = in_df.shape[0]

    df_gen.n = in_df.shape[0]

    df_gen._set_index_array()

    df_gen.directory = '' # since we have the full path

    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))

    return df_gen
train_gen = flow_from_dataframe(core_idg, train_df, 

                             path_col = 'path',

                            y_col = 'path', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 32)



valid_gen = flow_from_dataframe(core_idg, valid_df, 

                             path_col = 'path',

                            y_col = 'path', 

                            target_size = IMG_SIZE,

                             color_mode = 'rgb',

                            batch_size = 32) # we can use much larger batches for evaluation
t_x, t_y = next(train_gen)

fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))

for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):

    c_ax.imshow(c_x[:,:,0], cmap = 'bone', vmin = -127, vmax = 127)

    c_ax.set_title('{}'.format(c_y))

    c_ax.axis('off')
from keras.applications.inception_v3 import InceptionV3

base_iv3_model = InceptionV3(input_shape =  t_x.shape[1:], include_top = False, weights = 'imagenet')

base_iv3_model.trainable = False

from keras.applications.vgg16 import VGG16

base_vgg_model = VGG16(input_shape =  t_x.shape[1:], include_top = False, weights = 'imagenet')

base_vgg_model.trainable = False
from tqdm import tqdm

import pandas as pd

import h5py

from tqdm import tqdm

def write_df_as_hdf(out_path,

                    out_df,

                    compression='gzip'):

    with h5py.File(out_path, 'w') as h:

        for k, arr_dict in tqdm(out_df.to_dict().items()):

            try:

                s_data = np.stack(arr_dict.values(), 0)

                try:

                    h.create_dataset(k, data=s_data, compression=

                    compression)

                except TypeError as e:

                    try:

                        h.create_dataset(k, data=s_data.astype(np.string_),

                                         compression=compression)

                    except TypeError as e2:

                        print('%s could not be added to hdf5, %s' % (

                            k, repr(e), repr(e2)))

            except ValueError as e:

                print('%s could not be created, %s' % (k, repr(e)))

                all_shape = [np.shape(x) for x in arr_dict.values()]

                warn('Input shapes: {}'.format(all_shape))

                

def create_df(in_gen, 

              model,

              iters = 500):

    out_feature_vec = []

    out_paths = []

    for _, (c_x, c_y) in zip(tqdm(range(iters)), in_gen):

        out_feature_vec += [model.predict(c_x)]

        out_paths += [c_y]

    out_df = pd.DataFrame(dict(feature_vec = [x for x in np.concatenate(out_feature_vec,0)],

                               path = [p for p in np.concatenate(out_paths, 0)]))

    return pd.merge(out_df, age_df, on=['path'])
write_df_as_hdf('train.h5', 

                create_df(train_gen, base_vgg_model, 100)

               )
# check the file

with h5py.File('train.h5', 'r') as f:

    for k in f.keys():

        print(k, f[k].shape, f[k].dtype)
write_df_as_hdf('test.h5', 

                create_df(valid_gen, base_vgg_model, 100)

               )
!rm -rf ~/.keras