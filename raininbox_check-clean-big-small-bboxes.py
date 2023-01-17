import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

import ast
train = pd.read_csv('../input/global-wheat-detection/train.csv')

train[['x', 'y', 'w', 'h']] = pd.DataFrame(np.stack(train['bbox'].apply(lambda x: ast.literal_eval(x)))).astype(np.float32)

train['x1'] = train['x'] + train['w']

train['y1'] = train['y'] + train['h']

train['area'] = train['w'] * train['h']
for i in range(95, 100, 1):

    perc = np.percentile(train['area'], i)

    print(f"{i} percentile of area is {perc}")
for i in range(0, 5, 1):

    perc = np.percentile(train['area'], i)

    print(f"{i} percentile of area is {perc}")
Train_Box = train[train['area']<100]

Train_Box = Train_Box.sort_values(axis=0, ascending=True, by=['area'])

Train_Box = Train_Box.tail(4)

Train_Box.head()
grid_width = 2

grid_height = 2

images_id = ['6284044ed','ad256655b', '233cb8750', '6a8522f06']

bbox_id = [36287, 40034, 114998, 119089]

fig, axs = plt.subplots(grid_height, grid_width,

                        figsize=(15, 15))



for i, (img_id, box) in enumerate(zip(images_id, bbox_id)):

    ax = axs[int(i / grid_width), i % grid_width]

    image = cv2.imread(f'../input/global-wheat-detection/train/{img_id}.jpg', cv2.IMREAD_COLOR)

    box = [int(Train_Box['x'][box]),int(Train_Box['y1'][box]),int(Train_Box['x1'][box]),int(Train_Box['y'][box])]

    cv2.rectangle(image,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    ax.imshow(image.squeeze())



plt.show()
Train_Box = train[train['area']<200]

Train_Box = Train_Box.sort_values(axis=0, ascending=True, by=['area'])

Train_Box = Train_Box.tail(4)

Train_Box.head()
grid_width = 2

grid_height = 2

images_id = ['78752f185','71b460a14', '3f8f6b1a1', 'd0ab06fc3']

bbox_id = [115412, 4128, 145578, 66114]

fig, axs = plt.subplots(grid_height, grid_width,

                        figsize=(15, 15))



for i, (img_id, box) in enumerate(zip(images_id, bbox_id)):

    ax = axs[int(i / grid_width), i % grid_width]

    image = cv2.imread(f'../input/global-wheat-detection/train/{img_id}.jpg', cv2.IMREAD_COLOR)

    box = [int(Train_Box['x'][box]),int(Train_Box['y1'][box]),int(Train_Box['x1'][box]),int(Train_Box['y'][box])]

    cv2.rectangle(image,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    ax.imshow(image.squeeze())



plt.show()
Train_Box = train[train['area']<300]

Train_Box = Train_Box.sort_values(axis=0, ascending=True, by=['area'])

Train_Box = Train_Box.tail(4)

Train_Box.head()
grid_width = 2

grid_height = 2

images_id = ['0b2967a7a','1e9ff110c', 'c1577d6ff', '217c8fd61']

bbox_id = [112057, 3465, 119930, 125886]

fig, axs = plt.subplots(grid_height, grid_width,

                        figsize=(15, 15))



for i, (img_id, box) in enumerate(zip(images_id, bbox_id)):

    ax = axs[int(i / grid_width), i % grid_width]

    image = cv2.imread(f'../input/global-wheat-detection/train/{img_id}.jpg', cv2.IMREAD_COLOR)

    box = [int(Train_Box['x'][box]),int(Train_Box['y1'][box]),int(Train_Box['x1'][box]),int(Train_Box['y'][box])]

    cv2.rectangle(image,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    ax.imshow(image.squeeze())



plt.show()
Train_Box = train[train['area']<400]

Train_Box = Train_Box.sort_values(axis=0, ascending=True, by=['area'])

Train_Box = Train_Box.tail(4)

Train_Box.head()
grid_width = 2

grid_height = 2

images_id = ['060543bbf','94ea18562', 'bbce58f71', '408013a9d']

bbox_id = [88815, 84884, 122802, 112357]

fig, axs = plt.subplots(grid_height, grid_width,

                        figsize=(15, 15))



for i, (img_id, box) in enumerate(zip(images_id, bbox_id)):

    ax = axs[int(i / grid_width), i % grid_width]

    image = cv2.imread(f'../input/global-wheat-detection/train/{img_id}.jpg', cv2.IMREAD_COLOR)

    box = [int(Train_Box['x'][box]),int(Train_Box['y1'][box]),int(Train_Box['x1'][box]),int(Train_Box['y'][box])]

    cv2.rectangle(image,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    ax.imshow(image.squeeze())



plt.show()
Train_Box = train[train['area']<500]

Train_Box = Train_Box.sort_values(axis=0, ascending=True, by=['area'])

Train_Box = Train_Box.tail(4)

Train_Box.head()
grid_width = 2

grid_height = 2

images_id = ['1f255e0c5','ea88fb8ec', 'b7c9166b6', '9a50eab86']

bbox_id = [65825, 63607, 124682, 63881]

fig, axs = plt.subplots(grid_height, grid_width,

                        figsize=(15, 15))



for i, (img_id, box) in enumerate(zip(images_id, bbox_id)):

    ax = axs[int(i / grid_width), i % grid_width]

    image = cv2.imread(f'../input/global-wheat-detection/train/{img_id}.jpg', cv2.IMREAD_COLOR)

    box = [int(Train_Box['x'][box]),int(Train_Box['y1'][box]),int(Train_Box['x1'][box]),int(Train_Box['y'][box])]

    cv2.rectangle(image,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    ax.imshow(image.squeeze())



plt.show()
Train_Box = train[train['area']>27456.23999999996]

Train_Box = Train_Box.sort_values(axis=0, ascending=True, by=['area'])

Train_Box = Train_Box.head(4)

Train_Box.head()
grid_width = 2

grid_height = 2

images_id = ['9858d67dc','ffc870198', '536ef8d03', '3b552c95a']

bbox_id = [33828, 4793, 54363, 120167]

fig, axs = plt.subplots(grid_height, grid_width,

                        figsize=(15, 15))



for i, (img_id, box) in enumerate(zip(images_id, bbox_id)):

    ax = axs[int(i / grid_width), i % grid_width]

    image = cv2.imread(f'../input/global-wheat-detection/train/{img_id}.jpg', cv2.IMREAD_COLOR)

    box = [int(Train_Box['x'][box]),int(Train_Box['y1'][box]),int(Train_Box['x1'][box]),int(Train_Box['y'][box])]

    cv2.rectangle(image,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    ax.imshow(image.squeeze())



plt.show()
Train_Box = train[train['area']>27456.23999999996]

Train_Box = Train_Box.sort_values(axis=0, ascending=True, by=['area'])

Train_Box = Train_Box.tail(700)

Train_Box.head()
grid_width = 2

grid_height = 2

images_id = ['c4dc3c575','93d67b171', '93d67b171', 'd89f4ea06']

bbox_id = [42775, 4360, 4352, 33604]

fig, axs = plt.subplots(grid_height, grid_width,

                        figsize=(15, 15))



for i, (img_id, box) in enumerate(zip(images_id, bbox_id)):

    ax = axs[int(i / grid_width), i % grid_width]

    image = cv2.imread(f'../input/global-wheat-detection/train/{img_id}.jpg', cv2.IMREAD_COLOR)

    box = [int(Train_Box['x'][box]),int(Train_Box['y1'][box]),int(Train_Box['x1'][box]),int(Train_Box['y'][box])]

    cv2.rectangle(image,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    ax.imshow(image.squeeze())



plt.show()
Train_Box = train[train['area']>27456.23999999996]

Train_Box = Train_Box.sort_values(axis=0, ascending=True, by=['area'])

Train_Box = Train_Box.tail(350)

Train_Box.head()
grid_width = 2

grid_height = 2

images_id = ['be11c4e40','b8ddb6c73', '5a76259a0', '73ed5eb37']

bbox_id = [2817, 128030, 116083, 90575]

fig, axs = plt.subplots(grid_height, grid_width,

                        figsize=(15, 15))



for i, (img_id, box) in enumerate(zip(images_id, bbox_id)):

    ax = axs[int(i / grid_width), i % grid_width]

    image = cv2.imread(f'../input/global-wheat-detection/train/{img_id}.jpg', cv2.IMREAD_COLOR)

    box = [int(Train_Box['x'][box]),int(Train_Box['y1'][box]),int(Train_Box['x1'][box]),int(Train_Box['y'][box])]

    cv2.rectangle(image,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    ax.imshow(image.squeeze())



plt.show()
Train_Box = train[train['area']>27456.23999999996]

Train_Box = Train_Box.sort_values(axis=0, ascending=True, by=['area'])

Train_Box = Train_Box.tail(175)

Train_Box.head()
grid_width = 2

grid_height = 2

images_id = ['2c836cccb','d5943ea17', 'dcafcae79', '1e58125ec']

bbox_id = [123482, 54427, 52733, 38737]

fig, axs = plt.subplots(grid_height, grid_width,

                        figsize=(15, 15))



for i, (img_id, box) in enumerate(zip(images_id, bbox_id)):

    ax = axs[int(i / grid_width), i % grid_width]

    image = cv2.imread(f'../input/global-wheat-detection/train/{img_id}.jpg', cv2.IMREAD_COLOR)

    box = [int(Train_Box['x'][box]),int(Train_Box['y1'][box]),int(Train_Box['x1'][box]),int(Train_Box['y'][box])]

    cv2.rectangle(image,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    ax.imshow(image.squeeze())



plt.show()
Train_Box = train[train['area']>27456.23999999996]

Train_Box = Train_Box.sort_values(axis=0, ascending=True, by=['area'])

Train_Box = Train_Box.tail(15)

Train_Box.head(15)
grid_width = 3

grid_height = 5

images_id = ['b8ddb6c73', 'f1a8585e0', '51f2e0a05', '69fc3d3ff', '9adbfe503', '41c0123cc','a1321ca95', 'ad6e9eea2', '9a30dd802', 'd7a02151d', '409a8490c', '2cc75e9f5', 'a1321ca95', 'd067ac2b1', '42e6efaaa']

bbox_id = [128028, 53790, 53930, 1259, 54892, 173, 2169, 54702, 52868, 118211, 117344, 3687, 2159, 121633, 113947]

fig, axs = plt.subplots(grid_height, grid_width,

                        figsize=(15, 15))



for i, (img_id, box) in enumerate(zip(images_id, bbox_id)):

    ax = axs[int(i / grid_width), i % grid_width]

    image = cv2.imread(f'../input/global-wheat-detection/train/{img_id}.jpg', cv2.IMREAD_COLOR)

    box = [int(Train_Box['x'][box]),int(Train_Box['y1'][box]),int(Train_Box['x1'][box]),int(Train_Box['y'][box])]

    cv2.rectangle(image,

                  (box[0], box[1]),

                  (box[2], box[3]),

                  (220, 0, 0), 3)

    ax.set_title(img_id)

    ax.imshow(image.squeeze())



plt.show()
train = pd.read_csv('../input/global-wheat-detection/train.csv')

train[['x', 'y', 'w', 'h']] = pd.DataFrame(np.stack(train['bbox'].apply(lambda x: ast.literal_eval(x)))).astype(np.float32)

train['area'] = train['w'] * train['h']

train.shape
train_clean = train[train['area']>300]

train_clean = train_clean[train['w']>10]

train_clean = train_clean[train['h']>10]

train_clean = train_clean.drop([173,2169,118211,52868,117344,3687,2159,121633,113947])
print("remove {} boxes".format(train.shape[0] - train_clean.shape[0]))
train_clean.head()
train_clean.to_csv('train_clean.csv')