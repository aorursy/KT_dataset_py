import numpy as np

import pandas as pd

from pathlib import Path

import sys

import os

import pickle

from tqdm import tqdm_notebook as tqdm

from PIL import Image

import matplotlib.pyplot as plt

import seaborn as sns

import itertools
print(os.listdir('../input'))

print(os.listdir('../input/diabetic-retinopathy-detection-image-size'))
train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
len(train), len(test)
#the func is from https://www.kaggle.com/toshik/image-size-and-rate-of-new-whale

def get_size_list(targets, dir_target):

    result = list()

    for target in tqdm(targets):

        img = np.array(Image.open(os.path.join(dir_target, target+'.png')))

        result.append(img.shape)

    return result



# the func is from https://www.kaggle.com/kaerunantoka/extract-image-features

def get_size(file_name_list, dir_target):

    result = list()

    #filename = images_path + filename

    for file_name in tqdm(file_name_list):

        st = os.stat(f'{dir_target}/{file_name}.png')

        result.append(st.st_size)

    return result
train['image_shape'] = get_size_list(train.id_code.tolist(),

                                     dir_target='../input/aptos2019-blindness-detection/train_images')

test['image_shape'] = get_size_list(test.id_code.tolist(),

                                    dir_target='../input/aptos2019-blindness-detection/test_images')

train['image_size'] = get_size(train.id_code.tolist(),

                               dir_target='../input/aptos2019-blindness-detection/train_images')

test['image_size'] = get_size(test.id_code.tolist(),

                              dir_target='../input/aptos2019-blindness-detection/test_images')
for df in [train, test]:

    df['height'] = df['image_shape'].apply(lambda x:x[0])

    df['width'] = df['image_shape'].apply(lambda x:x[1])

    df['width_height_ratio'] = df['height'] / df['width']

    df['width_height_added'] = df['height'] + df['width']
train.head()
train.describe()
test.describe()
fig = plt.figure(figsize=(16,10))

plt.subplot(241)

plt.hist(train['width'])

plt.title("train width")

plt.xlim(200, 4500)



plt.subplot(242)

plt.hist(test['width'])

plt.title("test width")

plt.xlim(200, 4500)



plt.subplot(243)

plt.hist(train['height'])

plt.title("train height")

plt.xlim(200, 3100)



plt.subplot(244)

plt.hist(test['height'])

plt.title("test height")

plt.xlim(200, 3100)



plt.subplot(245)

plt.hist(train['width_height_ratio'])

plt.title("train width height ratio")

plt.xlim(0.6, 1.05)





plt.subplot(246)

plt.hist(test['width_height_ratio'])

plt.title("test width height ratio")

plt.xlim(0.6, 1.05)



plt.subplot(247)

plt.hist(train['width_height_added'])

plt.title("train width height added")



plt.subplot(248)

plt.hist(test['width_height_added'])

plt.title("train width height added");
sns.heatmap(train.corr(), cmap=plt.cm.Blues, annot=True);
train_meta = train.groupby(['width', 'height', 'diagnosis']).agg({'diagnosis':'count'}).unstack('diagnosis').fillna(0)

train_meta.columns = [f'{i[0]}_{i[1]}' for i in train_meta.columns]

train_meta['train_count'] = train_meta.sum(axis=1)



test_meta = test.groupby(['width', 'height']).agg({'id_code':'count'}).rename(columns={'id_code':'pub_test_count'})

count_ratio = train_meta.join(test_meta, how='outer')



for i in range(5):

    count_ratio.loc[:, f'{i}_ratio'] = count_ratio.iloc[:, i] / count_ratio['train_count']



count_ratio = count_ratio.fillna(0)



count_ratio = count_ratio.astype({'diagnosis_0': int, 'diagnosis_1': int, 'diagnosis_2': int,

                                  'diagnosis_3': int, 'diagnosis_4': int})

count_ratio = count_ratio.astype({'train_count': int, 'pub_test_count': int})



count_ratio.reset_index(inplace=True)

count_ratio.set_index(['width', 'height', 'train_count', 'pub_test_count'], inplace=True)
count_ratio
def im_show(height, width, num):

    tmp = train[(train['width'] == width) & (train['height'] == height)].id_code

    dir_target = '../input/aptos2019-blindness-detection/train_images'

    id = tmp.values[num]

    img = Image.open(os.path.join(dir_target, id +'.png'))

    plt.imshow(img.resize((256, 256)))

    plt.tick_params(bottom=False,

                    left=False,

                    right=False,

                    top=False,

                    labelbottom=False,

                    labelleft=False,

                    labelright=False,

                    labeltop=False)

    value = train.loc[train['id_code'] == id, :].values[0]

    plt.title(f'({value[4]},{value[5]})->(256,256)\n {id}, diagnosis:{value[1]}')



def five_img_plot(height, width):

    print('-' * 10)

    print(f'shape({height}, {width})')

    plt.figure(figsize=(16, 4))

    for i in range(5):

        plt.subplot(1,5,i+1)

        im_show(height, width, i)

    plt.show()
five_img_plot(480, 640)

five_img_plot(614, 819)

five_img_plot(1050, 1050)

five_img_plot(1536, 2048)

five_img_plot(1736, 2416)

five_img_plot(1958, 2588)

five_img_plot(2588, 3388)
pre_train = pd.read_csv('../input/diabetic-retinopathy-detection-image-size/pre_train_shape.csv')

pre_test = pd.read_csv('../input/diabetic-retinopathy-detection-image-size/pre_test_shape.csv')



for df in [pre_train, pre_test]:

    df['width_height_ratio'] = df['height'] / df['width']

    df['width_height_added'] = df['height'] + df['width']
len(pre_train), len(pre_test)
pre_train.head()
pre_test.head()
pre_train.describe()
fig = plt.figure(figsize=(16,10))

plt.subplot(241)

plt.hist(pre_train['width'])

plt.title("pre train width")

plt.xlim(200, 5500)



plt.subplot(242)

plt.hist(pre_test['width'])

plt.title("pre test width")

plt.xlim(200, 5500)



plt.subplot(243)

plt.hist(pre_train['height'])

plt.title("pre train height")

plt.xlim(200, 4000)



plt.subplot(244)

plt.hist(pre_test['height'])

plt.title("pre test height")

plt.xlim(200, 4000)



plt.subplot(245)

plt.hist(pre_train['width_height_ratio'])

plt.title("pre train width height ratio")

plt.xlim(0.6, 1.05)





plt.subplot(246)

plt.hist(pre_test['width_height_ratio'])

plt.title("pre test width height ratio")

plt.xlim(0.6, 1.05)



plt.subplot(247)

plt.hist(pre_train['width_height_added'])

plt.title("pre train width height added")



plt.subplot(248)

plt.hist(pre_test['width_height_added'])

plt.title("pre train width height added");
pre_train.drop('channel', axis=1, inplace=True)

pre_test.drop('channel', axis=1, inplace=True)
plt.rcParams["font.size"] = 14

# pre_train.rename(columns={'level': 'diagnosis'}, inplace=True)

plt.figure(figsize=(16,8))

plt.subplot(121)

sns.heatmap(pre_train.corr(), cmap=plt.cm.Blues, annot=True)

plt.title('previous_competition')



plt.subplot(122)

sns.heatmap(train.corr(), cmap=plt.cm.Blues, annot=True)

plt.title('this_competition')



plt.tight_layout()
pre_train_meta = pre_train.groupby(['width', 'height', 'level']).agg({'level':'count'}).unstack('level').fillna(0)

pre_train_meta.columns = [f'{i[0]}_{i[1]}' for i in pre_train_meta.columns]

pre_train_meta['train_count'] = pre_train_meta.sum(axis=1)



pre_test_meta = pre_test.groupby(['width', 'height']).agg({'image':'count'}).rename(columns={'image':'pub_test_count'})

pre_count_ratio = pre_train_meta.join(pre_test_meta, how='outer')



for i in range(5):

    pre_count_ratio.loc[:, f'{i}_ratio'] = pre_count_ratio.iloc[:, i] / pre_count_ratio['train_count']



pre_count_ratio = pre_count_ratio.fillna(0)



pre_count_ratio = pre_count_ratio.astype({'level_0': int, 'level_1': int, 'level_2': int, 'level_3': int, 'level_4': int})

pre_count_ratio = pre_count_ratio.astype({'train_count': int, 'pub_test_count': int})



pre_count_ratio.reset_index(inplace=True)

pre_count_ratio.set_index(['width', 'height', 'train_count', 'pub_test_count'], inplace=True)
pre_count_ratio
plt.rcParams["font.size"] = 13

plt.figure(figsize=(12, 8))

sns.heatmap(pre_count_ratio.iloc[:, 5:], cmap=plt.cm.Blues)

plt.xlabel('target')

plt.ylabel('width - height - number_of_train - number_of_public_test')

plt.title('Number of image and ratio by image shape in previous competition')

plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], ['0',  '1', '2', '3', '4'])

for i, j in itertools.product(range(5), range(len(pre_count_ratio))):

    train_count = pre_count_ratio.index[j][2]

    if train_count != 0:

        ratio = np.int(np.round(pre_count_ratio.iloc[j, i+5] * 100))

        count = pre_count_ratio.iloc[j, i]

        plt.text(i+0.2, j+0.8, f'{count:>4}', color='k'if ratio < 65 else "w")

        plt.text(i+0.5, j+0.8, f'{ratio:>3}%', color='k'if ratio < 65 else "w")

    elif train_count == 0:

        plt.text(i+0.5, j+0.8, '-', color='k')

plt.show()



plt.figure(figsize=(12, 8))

sns.heatmap(count_ratio.iloc[:, 5:], cmap=plt.cm.Blues)

plt.xlabel('target')

plt.ylabel('width - height - number_of_train - number_of_public_test')

plt.title('Number of image and ratio by image shape in present competition')

for i, j in itertools.product(range(5), range(len(count_ratio))):

    train_count = count_ratio.index[j][2]

    if train_count != 0:

        ratio = np.int(np.round(count_ratio.iloc[j, i+5] * 100))

        count = count_ratio.iloc[j, i]

        plt.text(i+0.2, j+0.8, f'{count:>4}', color='k'if ratio < 65 else "w")

        plt.text(i+0.5, j+0.8, f'{ratio:>3}%', color='k'if ratio < 65 else "w")

    elif train_count == 0:

        plt.text(i+0.5, j+0.8, '-', color='k')

    plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5], ['0',  '1', '2', '3', '4'])

plt.show();