# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import os
import glob
import cv2 
from math import ceil

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
train_df = pd.read_csv('../input/landmark-retrieval-2020/train.csv')
train_df.head()
train_list = glob.glob('../input/landmark-retrieval-2020/train/*/*/*/*')
test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')
index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')
len(train_list)
len(test_list)
len(index_list)
all_ids = train_df.landmark_id.unique()
len(all_ids)
all_ids
def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=ceil(len(img_matrix_list) / ncols), ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
num_of_samples = 3

# train_df.loc[train_df['id']=='a00009431492c304']
# we will be using this landmark and its corresponding id 14112 for visualizations

landmark_14112 = train_df.query(f'landmark_id == {14112}').sample(num_of_samples)['id']
landmark_14112
landmark_14112_image_list = []
title_list = [1,2,3]
for i, img in enumerate(landmark_14112):
    arg_img = int(np.argwhere(list(map(lambda x: img in x, train_list))).ravel())
    landmark_14112_img = cv2.imread(train_list[arg_img])[:,:,::-1]
    landmark_14112_image_list.append(landmark_14112_img)

plot_multiple_img(landmark_14112_image_list, title_list, ncols=3, main_title="")
def conv_horizontal(img):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    kernel = np.ones((3,3), np.float32)
    kernel[1] = np.array([0,0,0],np.float32)
    kernel[2] = np.array([-1,-1,-1],np.float32)
    conv = cv2.filter2D(img, -1, kernel)
    ax[0].imshow(img)
    ax[0].set_title('Original Image', fontsize=24)
    ax[1].imshow(conv)
    ax[1].set_title('Convolved Image', fontsize=24)
    plt.show()

def conv_vertical(img):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    kernel = np.ones((3,3), np.float32)
    kernel[0] = np.array([1,0,-1])
    kernel[1] = np.array([1,0,-1])
    kernel[2] = np.array([1,0,-1])
    conv = cv2.filter2D(img, -1, kernel)
    ax[0].imshow(img)
    ax[0].set_title('Original Image', fontsize=24)
    ax[1].imshow(conv)
    ax[1].set_title('Convolved Image', fontsize=24)
    plt.show()
conv_horizontal(landmark_14112_image_list[0])
conv_horizontal(landmark_14112_image_list[1])
conv_horizontal(landmark_14112_image_list[2])

conv_vertical(landmark_14112_image_list[0])
conv_vertical(landmark_14112_image_list[1])
conv_vertical(landmark_14112_image_list[2])


