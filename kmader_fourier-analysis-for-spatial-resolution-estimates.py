import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from skimage.io import imread # read image

from PIL import Image 

# imread fails on some of the tiffs so we use PIL

pil_imread = lambda c_file: np.array(Image.open(c_file)) 

from skimage.exposure import equalize_adapthist

from glob import glob



%matplotlib inline

import matplotlib.pyplot as plt
list_train = glob(os.path.join('..', 'input', 'train', '*', '*.jpg'))

print('Train Files found', len(list_train), 'first file:', list_train[0])

list_test = glob(os.path.join('..', 'input', '*', '*.tif'))

print('Test Files found', len(list_test), 'first file:', list_test[0])
from sklearn.preprocessing import LabelEncoder

def get_class_from_path(filepath):

    return os.path.dirname(filepath).split(os.sep)[-1]

full_train_df = pd.DataFrame([{'path': x, 'category': get_class_from_path(x)} for x in list_train])

cat_encoder = LabelEncoder()

cat_encoder.fit(full_train_df['category'])

nclass = cat_encoder.classes_.shape[0]

full_train_df.sample(3)
fig, ax1 = plt.subplots(1,1,figsize = (8, 6))

ax1.hist(cat_encoder.transform(full_train_df['category']), np.arange(nclass+1))

ax1.set_xticks(np.arange(nclass))

_ = ax1.set_xticklabels(cat_encoder.classes_, rotation = 45)
def imread_and_normalize(im_path):

    img_data = pil_imread(im_path)

    return img_data/255.0



test_img = imread_and_normalize(full_train_df['path'].values[0])

plt.imshow(test_img)
from numpy.fft import fft2

from scipy import signal

def gen_nd_psd(in_img, n_pts):

    out_f = np.linspace(0, 0.5, n_pts)

    out_psd = np.zeros((out_f.shape[0], 3))

    for i in range(in_img.shape[2]):

        for j in range(in_img.shape[1]):

            f, nPxx_den = signal.periodogram(in_img[:,j, i], 1, 

                                            'flattop', 

                                            scaling='density')

            if j==0:

                Pxx_den = nPxx_den

            else:

                Pxx_den += nPxx_den

        Pxx_den = Pxx_den/in_img.shape[1]

        out_psd[:, i] = np.interp(out_f,f, Pxx_den)

    return out_f, out_psd
%%time

out_f, out_psd = gen_nd_psd(test_img, 100)
fig, rgb_ax = plt.subplots(1,3, figsize = (12, 3))

for i, c_ax in enumerate(rgb_ax):

    c_ax.semilogy(out_f, out_psd[:, i], '.')

    c_ax.set_ylim([1e-9, 1e2])

    c_ax.set_title('RGB'[i]+' color information')

    c_ax.set_xlabel('frequency [Hz]')

    c_ax.set_ylabel('PSD [V**2/Hz]')
plt.plot(np.log10(out_psd))
%%time

subset_df = full_train_df.groupby('category').apply(lambda x: x.sample(1)).reset_index(drop = True)

subset_df['img'] = subset_df['path'].map(lambda x: imread_and_normalize(x))

subset_df['psd'] = subset_df['img'].map(lambda x: gen_nd_psd(x, 100)[1])
fig, c_axs = plt.subplots(2, subset_df.shape[0], figsize = (24, 6))

for (c_ax, m_ax), (_, c_row) in zip(c_axs.T, subset_df.iterrows()):

    c_ax.imshow(c_row['img'])

    c_ax.set_title(c_row['category'])

    c_ax.axis('off')

    m_ax.plot(np.log10(c_row['psd']))

    m_ax.set_ylim(-5, 0)
fig, ax = plt.subplots(1,1, figsize = (6, 6))

for _, c_row in subset_df.iterrows():

    ax.plot(np.log10(c_row['psd'][:, 0]), label = c_row['category'])

ax.legend()
fig, ax = plt.subplots(1,1, figsize = (6, 6))

for _, c_row in subset_df.iterrows():

    ax.plot(np.log10(c_row['psd'][10:80, 0]), label = c_row['category'])

ax.legend()
%%time

bigger_subset_df = full_train_df.groupby('category').apply(lambda x: x.sample(30)).reset_index(drop = True)

bigger_subset_df['img'] = bigger_subset_df['path'].map(lambda x: imread_and_normalize(x))

bigger_subset_df['psd'] = bigger_subset_df['img'].map(lambda x: gen_nd_psd(x, 100)[1])
d_gen = generate_even_batch(full_train_df)

for _, (x, y) in zip(range(1), d_gen):

    print(x.shape, y.shape)
%%time

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(full_train_df, 

                                     test_size = 0.15,

                                    random_state = 2018,

                                    stratify = full_train_df['category'])

print('Train', train_df.shape[0], 'Test', test_df.shape[0])

train_gen = generate_even_batch(train_df, 3, chunk_count = 20)

test_gen = generate_even_batch(test_df, 10, chunk_count = 30)

# cache the test_gen_data

(test_x, test_y) = next(test_gen)

print('Test Data', test_x.shape)
from tqdm import tqdm

out_dict_list = []

for c_file in tqdm(list_test):

    ck_data = read_chunk(c_file, n_chunk = 100)

    ck_pred = model.predict(ck_data)

    # take the average prediction

    mean_vec = np.mean(ck_pred,0)

    out_dict_list += [{

        'fname': os.path.basename(c_file),

        'camera': np.argmax(mean_vec,0)

    }]  
df = pd.DataFrame(out_dict_list)

df['camera'] = df['camera'].map(cat_encoder.inverse_transform)

df[['fname', 'camera']].to_csv("submission.csv", index=False)

df.sample(3)
fig, ax1 = plt.subplots(1,1,figsize = (8, 6))

ax1.hist(cat_encoder.transform(df['camera']), np.arange(nclass+1))

ax1.set_xticks(np.arange(nclass)+0.5)

_ = ax1.set_xticklabels(cat_encoder.classes_, rotation = 90)