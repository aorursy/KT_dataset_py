# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



current_path = os.getcwd()

data_path = os.path.join(current_path, '..', 'input', '2020backofwordrcv')

print(os.listdir(data_path))

# Any results you write to the current directory are saved as output.
train_path = os.path.join(data_path, 'traindata')

train_label_list = os.listdir(train_path)
train_label_info = {}



for label in train_label_list:

    _tmp_label_path = os.path.join(train_path, label)

    _num_images = len(os.listdir(_tmp_label_path))

    

    if label not in train_label_info.keys():

        train_label_info[label] = _num_images

        

num_show = 10

print('Total number of labels: {}'.format(len(train_label_info)))

print('Just show {} items {} ...'.format(num_show, list(train_label_info.keys())[:num_show]))
import matplotlib.pyplot as plt



plt.figure(figsize=(20, 5))

plt.bar(range(len(train_label_info)), list(train_label_info.values()))

plt.show()
import random



num_label = len(train_label_info)

num_image = train_label_info[train_label_list[0]]
fig = plt.figure(figsize=(18, 18))



for i in range(9):

    ax = fig.add_subplot(3, 3, i + 1)

    

    _random_label = random.randint(0, num_label - 1)

    _random_image = random.randint(0, num_image - 1)

    

    _label_path = os.path.join(train_path, train_label_list[_random_label])

    _image_list = os.listdir(_label_path)

    _image_path = os.path.join(_label_path, _image_list[_random_image])

    

    image = plt.imread(_image_path)

    plt.imshow(image)

    

    _title = '{}, {}'.format(train_label_list[_random_label], _random_image)

    ax.set_title(_title)

    ax.axis('off')
test_path = os.path.join(data_path, 'testdata')

test_image_list = os.listdir(test_path)



num_test_images = len(test_image_list)

print('Total number of test images: {}'.format(num_test_images))
fig = plt.figure(figsize=(18, 18))



for i in range(9):

    ax = fig.add_subplot(3, 3, i + 1)

    

    _random_image = random.randint(0, num_test_images - 1)

    _image_path = os.path.join(test_path, test_image_list[_random_image])

    

    image = plt.imread(_image_path)

    plt.imshow(image)

    

    _title = '{}'.format(test_image_list[_random_image])

    ax.set_title(_title)

    ax.axis('off')
test_label_path = os.path.join(data_path, "test_label.csv")

test_label = pd.read_csv(test_label_path)

print(test_label.head(5))
cum_h = []

cum_w = []



for label_list in train_label_list:

    _label_path = os.path.join(train_path, label_list)

    _image_list = os.listdir(_label_path)

    for image_file in _image_list:

        _image_path = os.path.join(_label_path, image_file)

        image = plt.imread(_image_path)



        h, w = image.shape[:2]

        cum_h.append(h)

        cum_w.append(w)



cum_h = np.array(cum_h)

cum_w = np.array(cum_w)

print('Avg. height:\t{:.1f}\nAvg. width:\t{:.1f}'.format(np.mean(cum_h), np.mean(cum_w)))

print('Median height:\t{}\nMedian width:\t{}'.format(np.median(cum_h), np.median(cum_w)))