%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 10)

plt.rcParams["figure.dpi"] = 125

plt.rcParams["font.size"] = 14

plt.rcParams['font.family'] = ['sans-serif']

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.style.use('ggplot')

sns.set_style("whitegrid", {'axes.grid': False})

from itertools import cycle

prop_cycle = plt.rcParams['axes.prop_cycle']

colors = prop_cycle.by_key()['color']
from pathlib import Path

import numpy as np

import pandas as pd

import doctest

import copy

import functools

import zipfile as zf

from skimage.io import imread

from skimage.util import montage as montage2d

from skimage.color import label2rgb

from functools import lru_cache

from shutil import copyfile

from tqdm import tqdm_notebook

# tests help notebooks stay managable



def autotest(func):

    globs = copy.copy(globals())

    globs.update({func.__name__: func})

    doctest.run_docstring_examples(

        func, globs, verbose=True, name=func.__name__)

    return func
base_path = Path('..') / 'input'

def read_minst_csv(in_path, src_cat):

    c_df = pd.read_csv(in_path).assign(src=in_path.parent.stem).assign(src_cat=src_cat)

    c_df.columns = ['label']+['pix_{:03d}'.format(i) for i in range(784)]+['src', 'src_cat']

    return c_df

all_input_dict = {c_path.parent.stem: read_minst_csv(c_path, idx) for idx, c_path in enumerate(base_path.glob('*/*train*.csv'))}

all_df = pd.concat(all_input_dict.values())

all_df.sample(3)
from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

from skimage.feature import canny

from skimage.filters import sobel



image_df = pd.DataFrame({

    'src': all_df['src'].values,

    'label': all_df['label'].map(lambda x: 9 if x==10 else x).values, # remap 10 to 9

    # run edge detection to make the images more similar

    'image': all_df.iloc[:, 1:-2].apply(lambda x: 

                                        np.expand_dims(sobel(np.reshape(x.values, (28, 28)).astype('float')), -1)

                                        , 1)

}).query('label<=9')





src_encoder = LabelEncoder()

image_df['src_cat'] = src_encoder.fit_transform(image_df['src'])

image_df['src_cat_oh'] = to_categorical(image_df['src_cat']).tolist()

image_df['label_oh'] = to_categorical(image_df['label']).tolist()

image_df.sample(3)



fig, m_axs = plt.subplots(3, 3)

for (_, c_row), c_ax in zip(image_df.sample(9).iterrows(), m_axs.flatten()):

    c_ax.imshow(c_row['image'][:, :, 0])

    c_ax.set_title('{src}-{label}'.format(**c_row))

    c_ax.axis('off')



image_df.sample(3)
image_df['image_mean'] = image_df['image'].map(np.mean)

image_df.groupby(['src', 'label']).agg({'image_mean': 'mean'})
train_df = image_df[image_df['src'].isin(['fashionmnist', 'sign-language-mnist'])].copy()

test_df = image_df[~image_df['src'].isin(['fashionmnist', 'sign-language-mnist'])].copy()

print(train_df.shape, test_df.shape)
from keras import layers, models

from keras.utils import vis_utils

from keras import backend as K

from IPython.display import SVG, display
def encoder_model():

    in_lay = layers.Input((28, 28, 1), name='Image_Input')

    x = layers.BatchNormalization()(in_lay)

    for i in range(3):

        x = layers.Conv2D(8, (3,3), activation='relu')(x)

        if i<2:

            x = layers.MaxPool2D((2,2))(x)

    x = layers.Flatten()(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Dense(16, name='Feature_Vector', activation='tanh')(x)

    return models.Model(inputs=[in_lay], outputs=[x], name='CNN_Encoder')

enc_model = encoder_model()

display(SVG(vis_utils.model_to_dot(enc_model, show_shapes=True).create_svg()))
def _triplet_func(x):

    a_min_b, a_min_c, b_min_c = x

    return K.mean(K.square(a_min_b), axis=1)-K.mean(K.square(a_min_c)+K.square(b_min_c), axis=1)/2

    

def create_triplet_model(enc_model):

    img_a = layers.Input((28, 28, 1), name='Image_Same_A')

    img_b = layers.Input((28, 28, 1), name='Image_Same_B')

    img_diff = layers.Input((28, 28, 1), name='Image_Different')

    feat_a = enc_model(img_a)

    feat_b = enc_model(img_b)

    feat_diff = enc_model(img_diff)

    a_min_b = layers.subtract([feat_a, feat_b])

    a_min_c = layers.subtract([feat_a, feat_diff])

    b_min_c = layers.subtract([feat_b, feat_diff])

    trip_score = layers.Lambda(_triplet_func, name='TripletLoss')([a_min_b, a_min_c, b_min_c])

    return models.Model(inputs=[img_a, img_b, img_diff], outputs=[trip_score])



triplet_model = create_triplet_model(enc_model)

display(SVG(vis_utils.model_to_dot(triplet_model, show_shapes=True).create_svg()))
def triplet_loss(y_true, y_pred):

    return y_pred

triplet_model.compile(optimizer='adam', loss=triplet_loss)
def gen_trip_batch(in_df):

    while True:

        for _, c_df in in_df.groupby('src'):

            label = c_df.sample(1)['label'].iloc[0]

            pos_rows = c_df[c_df['label'].isin([label])].sample(2)

            a_row, b_row = pos_rows.iloc[0], pos_rows.iloc[1]

            c_row = c_df[~c_df['label'].isin([label])].sample(1).iloc[0]

            yield {'Image_Same_A': a_row['image'], 'Image_Same_B': b_row['image'], 'Image_Different': c_row['image']}, {'TripletLoss': [0]}

fig, m_axs = plt.subplots(3, 3)

for c_axs, (t_x, t_y) in zip(m_axs, gen_trip_batch(train_df)):

    for c_ax, (k,v) in zip(c_axs, t_x.items()):

        c_ax.imshow(v[:, :, 0], cmap='gray')

        c_ax.set_title(k)

        c_ax.axis('off')
@autotest

def batch_it(in_gen, batch_size=64):

    """Collect and batch output from a generator

    >>> gen_obj = [({'x': [i]}, {'y': [4-i]}) for i in range(3)]

    >>> b_gen = batch_it(gen_obj, 2)

    >>> x, y = next(b_gen)

    >>> for k,v in x.items(): print(k, v) # doctest: +NORMALIZE_WHITESPACE

    x [[0]

     [1]]

    >>> for k,v in y.items(): print(k, v) # doctest: +NORMALIZE_WHITESPACE

    y [[4]

     [3]]

    """

    out_vals = []

    for c_vals in in_gen:

        out_vals += [c_vals]

        if len(out_vals)==batch_size:

            yield tuple([{k: np.stack([c_row[i][k] for c_row in out_vals], 0) 

                       for k in c_vals[i].keys()}

                       for i in range(len(c_vals))])

            out_vals = []
def fit_model(in_model, epochs=1, steps_per_epoch=100, validation_steps=10):

    return triplet_model.fit_generator(batch_it(gen_trip_batch(train_df)),

                           steps_per_epoch=steps_per_epoch,

                           validation_data=batch_it(gen_trip_batch(test_df)),

                           validation_steps=validation_steps,

                           epochs=epochs)



def show_triplet_batch(n=5):

    """Shows a batch of n images from all datasets"""

    fig, m_axs = plt.subplots(3, n, figsize=(12, 12))

    for c_axs, (t_x, t_y) in zip(m_axs.T, gen_trip_batch(image_df)):

        vec_out = {}

        for c_ax, (k,v) in zip(c_axs, t_x.items()):

            c_ax.imshow(v[:, :, 0], cmap='gray')

            vec_out[k] = enc_model.predict(np.expand_dims(v, 0))[0]

            c_ax.set_title(k)

            c_ax.axis('off')

        a_min_b = np.sqrt(np.mean(np.square(vec_out['Image_Same_A']-vec_out['Image_Same_B'])))

        a_min_c = np.sqrt(np.mean(np.square(vec_out['Image_Same_A']-vec_out['Image_Different'])))

        c_ax.set_title('A-B={:2.1%}\nA-C={:2.1%}'.format(a_min_b, a_min_c))

show_triplet_batch()
fit_model(triplet_model, 1)

show_triplet_batch()
fit_model(triplet_model, 9)

show_triplet_batch()
image_df['feature_vec'] = enc_model.predict(np.stack(image_df['image'].values, 0)).tolist()
from sklearn.decomposition import PCA

feat_pca = PCA(n_components=2)

feat_pca.fit(np.stack(image_df['feature_vec'].values, 0))
fig, m_axs = plt.subplots(1, 3, figsize=(10, 3))

for (c_src, src_df), c_ax in zip(image_df.groupby('src'), m_axs):

    for k, c_rows in src_df.groupby('label'):

        xy_vec = feat_pca.transform(np.stack(c_rows['feature_vec'], 0))

        c_ax.plot(xy_vec[:, 0], xy_vec[:, 1], '.', label='#{}'.format(k), ms=0.5, alpha=0.5)

    c_ax.set_title(c_src)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split

fig, m_axs = plt.subplots(1, 3, figsize=(15, 5))

for (c_src, src_df), c_ax in zip(image_df.groupby('src'), m_axs):

    src_train_df, src_test_df = train_test_split(src_df, random_state=2019, test_size=0.8)

    lr = LogisticRegression()

    lr.fit(np.stack(src_train_df['feature_vec'], 0), src_train_df['label'])

    pred_lr = lr.predict(np.stack(src_test_df['feature_vec'], 0))

    conf_mat = confusion_matrix(pred_lr, src_test_df['label'])

    sns.heatmap(conf_mat, ax=c_ax)

    c_ax.set_title('{} {:2.2%}'.format(c_src, accuracy_score(pred_lr, src_test_df['label'])))