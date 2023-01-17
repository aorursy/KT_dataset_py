import numpy as np

import json

import os

import matplotlib.pyplot as plt

%matplotlib inline



# print contents of input directory

print(os.listdir("../input"))
# load train set meta data 

train_anns_file = '../input/train2019.json'

with open(train_anns_file) as da:

    train_anns = json.load(da)
# print train set stats

print('Number of train images ' + str(len(train_anns['images'])))

print('Number of classes      ' + str(len(train_anns['categories'])))

category_ids = [cc['category_id'] for cc in train_anns['annotations']]

_, category_counts = np.unique(category_ids, return_counts=True)

plt.plot(np.sort(category_counts))

plt.title('classes sorted by amount of train images')

plt.xlabel('sorted class')

plt.ylabel('number of train images per class')

plt.show()
# display random image

rand_id = np.random.randint(len(train_anns['images']))

im_meta = train_anns['images'][rand_id]

im_category = train_anns['annotations'][rand_id]['category_id']

im = plt.imread('../input/train_val2019/' + im_meta['file_name'])

plt.imshow(im)

plt.title('image id: ' + str(rand_id) + ', class:' + str(im_category) + ', rights holder: ' + im_meta['rights_holder'])

plt.xticks([])

plt.yticks([])

plt.axis()

plt.show()