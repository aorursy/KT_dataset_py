import pandas as pd

import numpy as np

import os

from tqdm import tqdm

import cv2

import matplotlib.pyplot as plt

import time
FOLDER = '/kaggle/input/'

IMAGES = FOLDER + 'train_images/'

print(os.listdir(FOLDER))
df_train = pd.read_csv(FOLDER + 'train.csv')

df_train_idx = df_train.set_index("image_id")

idx_train = df_train['image_id']

unicode_map = {codepoint: char for codepoint, char in pd.read_csv(FOLDER + 'unicode_translation.csv').values}
def label_reader(label):

    try:

        code_arr = np.array(label['labels'].split(' ')).reshape(-1, 5)

    except:

        return

    return code_arr
idx = idx_train[0]

df_code = pd.DataFrame(label_reader(df_train_idx.loc[idx]))

df_code['image_id'] = idx

df_code.columns = ['char', 'x', 'y', 'w', 'h', 'image_id']

df_code[['x', 'y', 'w', 'h']] = df_code[['x', 'y', 'w', 'h']].astype('int')
def get_center(coord):

    return np.vstack([coord[:, 0] + coord[:, 2] //2, coord[:, 1] + coord[:, 3] //2]).T
coord = df_code.query('image_id == "{}"'.format(idx))[['x', 'y','w','h']].values

centers =get_center(coord)
image_path = IMAGES + idx + '.jpg'

img = cv2.imread(image_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.scatter(centers[:,0], centers[:,1])

plt.imshow(img)

plt.show()
from sklearn.cluster import KMeans
def get_cluster_n(centers, min_n=3, max_n=10):

    stds_list = []

    for n in range(min_n, max_n):

        X = centers.copy()

        X[:, 1] = X[:, 1]/100



        df_center = pd.DataFrame(centers)

        df_center['col_n'] = KMeans(n_clusters=n).fit(X).labels_

        stds_list.append(df_center.groupby('col_n').std().mean().values)



    stds = np.array(stds_list)

    xsm = np.log(stds[:,0])

    n_xsm = np.argmin(xsm[1:] - xsm[:-1]) + 1

    

    return n_xsm + min_n
get_cluster_n(centers)
n = get_cluster_n(centers)

X = centers.copy().astype('float')

X[:, 1] = X[:, 1]/100

df_center = pd.DataFrame(centers)

df_center['col_n'] = KMeans(n_clusters=n).fit(X).labels_

cols = df_center['col_n'].unique()

for col in cols:

    temp = df_center.query('col_n == {}'.format(col))

    plt.scatter(temp[0], temp[1])

    plt.imshow(img)

plt.show()
df_center['char'] = df_code.query('image_id == "{}"'.format(idx))['char'] # add unicode

cols = df_center.sort_values(0, ascending=False)['col_n'].unique() # sort by center_x because clustering labels are random.

chars = []

for col in cols:

    chars.extend(df_center.query('col_n == {}'.format(col)).sort_values(1)['char'].replace(unicode_map))

    chars.append(' ')
string = ''

for c in chars:

    string += c
string
def gen_df_code(df_idx, idx):

    df_code = pd.DataFrame(label_reader(df_idx.loc[idx]), columns = ['char', 'x', 'y', 'w', 'h'])

    df_code['image_id'] = idx

    df_code = df_code.reset_index()

    df_code[['x','y','w','h']] = df_code[['x','y','w','h']].astype('int')



    centers = get_center(df_code[['x','y','w','h']].values)

    df_code[['center_x', 'center_y']] = pd.DataFrame(centers)



    X = centers.copy().astype('float')

    X[:, 1] = X[:, 1]/100

    df_code['col_n'] =  KMeans(n_clusters=get_cluster_n(centers)).fit(X).labels_

    

    new_col_n = np.zeros(0)

    new_index = np.zeros(0)

    cols = df_code.sort_values('center_x', ascending=False)['col_n'].unique()

    for i, col in enumerate(cols):

        temp = df_code.query('col_n == {}'.format(col))

        new_index = np.hstack([new_index, temp['index'].values])

        new_col_n = np.hstack([new_col_n, np.ones(len(temp)) * i])



    del df_code['col_n']

    df_new_idx = pd.DataFrame([new_index, new_col_n]).T

    df_new_idx.columns = ['index', 'col_n']

    df_code = pd.merge(df_code, df_new_idx, on='index').sort_values('col_n').reset_index(drop=True)

    del df_code['index']

    df_code['col_n'] = df_code['col_n'].astype('int')



    return df_code
def gen_string(df_code):

    cols = df_code['col_n'].unique()

    chars = []

    for col in cols:

        chars.extend(df_code.query('col_n == {}'.format(col)).sort_values('center_y')['char'].replace(unicode_map))

        chars.append(' ')



    string = ''

    for c in chars:

        string += c



    print(string)
for idx in tqdm(idx_train[:40]):

    df_code = gen_df_code(df_train_idx, idx)

    gen_string(df_code)



    image_path = IMAGES + idx + '.jpg'

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cols = df_code['col_n'].unique()

    for col in cols:

        centers = df_code.query('col_n == {}'.format(col))[['center_x','center_y']].values

        plt.scatter(centers[:,0], centers[:,1])

    plt.imshow(img)

    plt.show()