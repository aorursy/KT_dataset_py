import numpy as np
import pandas as pd
import cv2
import re
from tqdm.notebook import tqdm
from PIL import Image
import hashlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()

DIR_INPUT = '../Wheat'
DIR_TRAIN_IMAGES = f'{DIR_INPUT}/train'
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
train_df.shape
train_df.head()
train_df['image_id'].nunique()
train_df['height'].value_counts(), train_df['width'].value_counts()
def calculate_hash(im):
    md5 = hashlib.md5()
    md5.update(np.array(im).tostring())
    
    return md5.hexdigest()
    
def get_image_meta(image_id, image_src, dataset='train'):
    im = Image.open(image_src)
    extrema = im.getextrema()
    
    meta = {
        'image_id': image_id,
        'dataset': dataset,
        'hash': calculate_hash(im),
        'r_min': extrema[0][0],
        'r_max': extrema[0][1],
        'g_min': extrema[1][0],
        'g_max': extrema[1][1],
        'b_min': extrema[2][0],
        'b_max': extrema[2][1],
        'height': im.size[0],
        'width': im.size[1],
        'format': im.format,
        'mode': im.mode
    }
    return meta
data = []

for i, image_id in enumerate(tqdm(train_df['image_id'].unique(), total=train_df['image_id'].unique().shape[0])):
    data.append(get_image_meta(image_id, DIR_TRAIN_IMAGES + '/{}.jpg'.format(image_id)))
meta_df = pd.DataFrame(data)
meta_df.head()
duplicates = meta_df.groupby(by='hash')[['image_id']].count().reset_index()
duplicates = duplicates[duplicates['image_id'] > 1]
duplicates.reset_index(drop=True, inplace=True)

duplicates = duplicates.merge(meta_df[['image_id', 'hash']], on='hash')

duplicates.head(20)
train_df['x'] = -1
train_df['y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)
train_df
train_df.groupby(by='image_id')['source'].count().agg(['min', 'max', 'mean'])
source = train_df['source'].value_counts()
source
fig = go.Figure(data=[go.Pie(labels=source.index, values=source.values)])

fig.update_layout(title='Source distribution')
fig.show()
def show_images(image_ids):
    
    col = 5
    row = min(len(image_ids) // col, 5)
    
    fig, ax = plt.subplots(row, col, figsize=(16, 8))
    ax = ax.flatten()

    for i, image_id in enumerate(image_ids):
        image = cv2.imread(DIR_TRAIN_IMAGES + '/{}.jpg'.format(image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax[i].set_axis_off()
        ax[i].imshow(image)
        ax[i].set_title(image_id)
        
def show_image_bb(image_data):
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    image = cv2.imread(DIR_TRAIN_IMAGES + '/{}.jpg'.format(image_data.iloc[0]['image_id']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i, row in image_data.iterrows():
        
        cv2.rectangle(image,
                      (int(row['x']), int(row['y'])),
                      (int(row['x']) + int(row['w']), int(row['y']) + int(row['h'])),
                      (220, 0, 0), 3)

    ax.set_axis_off()
    ax.imshow(image)
    ax.set_title(image_id)
show_images(train_df.sample(n=15)['image_id'].values)
show_image_bb(train_df[train_df['image_id'] == '5e0747034'])
show_image_bb(train_df[train_df['image_id'] == '5b13b8160'])
show_image_bb(train_df[train_df['image_id'] == '1f2b1a759'])