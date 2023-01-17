# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict

bad_pos_to_indices= defaultdict(list)

true_idx_2_shuffled_idx = {}

dev_scores = []

dev_pos_list = []

for i in range(num_dev):

    pred_dist_idx = list(np.argsort(dev_final_combined[i]))

    dev_pos = pred_dist_idx.index(i)

    dev_pos_list.append(dev_pos)

    if dev_pos < 20:

        dev_scores.append(1 / (dev_pos + 1))



    else:

        dev_scores.append(0.0)

        bad_pos_to_indices[dev_pos].append(dev_indexes[i])

        true_idx_2_shuffled_idx[dev_indexes[i]] = i



print("Development MAP@20:", np.mean(dev_scores))

print("Mean index of true image", np.mean(dev_pos_list))

print("Median index of true image", np.median(dev_pos_list))
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

check_index= 1377 # chech this index from bad_pos_to_indices 

img = mpimg.imread('/kaggle/input/cs5785-fall-2019-final/images_train/{}.jpg'.format(check_index))

imgplot = plt.imshow(img)

plt.show()

print("-------------------------")

print("Real Tags")

print(train_dev_tags[check_index])



predicted_tag_index= true_idx_2_shuffled_idx[check_index]

print("-------------------------")

print("Predicted Tags")

for index, val in enumerate(tag_pred_dev[predicted_tag_index]):

    if val == 1:

        print(idx2tags[index])

print("-------------------------")

print("Real Description")

print(train_dev_desc[check_index])