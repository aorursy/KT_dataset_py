import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split

%matplotlib inline
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
train.head()
train.inc_angle = train.inc_angle.apply(lambda x: np.nan if x == 'na' else x)
img1 = train.loc[20, ['band_1', 'band_2']]
# 1st 2nd and 3rd 

img1 = np.stack([img1['band_1'], img1['band_2']], -1).reshape(75, 75, 2)
#### band 1

plt.imshow(img1[:, :, 0] )
# band 2

plt.imshow(img1[:, :, 1])
combined = img1[:, :, 0] / img1[:, :, 1]
r = img1[:, :, 0]

r = (r + abs(r.min())) / np.max((r + abs(r.min())))



g = img1[:, :, 1]

g = (g + abs(g.min())) / np.max(g + abs(g.min()))



b = img1[:, :, 0] / img1[:, :, 1]

b = (((b) / np.max(b)) + abs((b) / np.max(b))) / np.max((((b) / np.max(b)) + abs((b) / np.max(b))))
plt.imshow(np.dstack((r, g, b)))
def color_composite(data):

    rgb_arrays = []

    for i, row in data.iterrows():

        band_1 = np.array(row['band_1']).reshape(75, 75)

        band_2 = np.array(row['band_2']).reshape(75, 75)

        band_3 = band_1 / band_2



        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))

        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))

        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))



        rgb = np.dstack((r, g, b))

        rgb_arrays.append(rgb)

    return np.array(rgb_arrays)
rgb_train = color_composite(train)
rgb_test = color_composite(test)
rgb_train.shape
rgb_test.shape
# look at random ships

print('Looking at random ships')

ships = np.random.choice(np.where(train.is_iceberg ==0)[0], 9)

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = rgb_train[ships[i], :, :]

    ax.imshow(arr)

    

plt.show()
# look at random icebergs

print('Looking at random icebergs')

icebergs = np.random.choice(np.where(train.is_iceberg ==1)[0], 9)

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = rgb_train[icebergs[i], :, :]

    ax.imshow(arr)

    

plt.show()
idx = np.random.choice(range(0, len(test)), 9)

test_img = color_composite(test.iloc[idx])



# look at random icebergs

print('Looking at random test images')

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = test_img[i, :, :]

    ax.imshow(arr)

    

plt.show()
os.makedirs('./data/composites', exist_ok= True)

os.makedirs('./data/composites/train', exist_ok=True)

os.makedirs('./data/composites/valid', exist_ok=True)

os.makedirs('./data/composites/test', exist_ok=True)



train_y, test_y = train_test_split(train.is_iceberg, test_size=0.3)

train_index, test_index = train_y.index, test_y.index



#save train images

for idx in train_index:

    img = rgb_train[idx]

    plt.imsave('./data/composites/train/' + str(idx) + '.jpg',  img)



#save valid images

for idx in test_index:

    img = rgb_train[idx]

    plt.imsave('./data/composites/valid/' + str(idx) + '.jpg',  img)



#save test images

for idx in range(len(test)):

    img = rgb_test[idx]

    plt.imsave('./data/composites/test/' + str(idx) + '.jpg',  img)