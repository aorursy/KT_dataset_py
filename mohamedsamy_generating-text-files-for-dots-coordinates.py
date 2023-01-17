# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import cv2

import matplotlib.pyplot as plt

import time

import pickle



plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'none'

plt.rcParams['image.cmap'] = 'gray'

import glob

import os

from sklearn.cluster import KMeans

# import caffe

%matplotlib inline
#train_data = pd.read_csv('../input/Train/train.csv')

#train_imgs = sorted(glob.glob('../input/Train/*.jpg'), key=lambda name: int(os.path.basename(name)[:-4]))

train_df = pd.read_csv('../input/Train/train.csv')

lion_classes = ['adult_males', 'subadult_males', 'adult_females', 'juveniles', 'pups']

#train_df.shape[0]

for img_fn in range(10):

    img_fn = str(img_fn)

    print('Image: ', img_fn)

    for lion_class_idx, lion_class_str in enumerate(lion_classes):

        if train_df[lion_class_str][int(img_fn)] <=0:

            continue

        img=cv2.imread("../input/TrainDotted/"+img_fn+".jpg")

        img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        

        if lion_class_idx == 0:

            lower_red = np.array([0,(95-5)*2.55,(95-5)*2.55])

            upper_red = np.array([2,(95+5)*2.55,(95+5)*2.55])

            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

            lower_red = np.array([178,(95-5)*2.55,(95-5)*2.55])

            upper_red = np.array([180,(95+5)*2.55,(95+5)*2.55])

            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

            red_mask = mask0+mask1

            mask = red_mask

        

        elif lion_class_idx == 1:

            lower_magenta = np.array([150,(95-5)*2.55,(95-5)*2.55])

            upper_magenta = np.array([155,(95+5)*2.55,(95+5)*2.55])

            magenta_mask = cv2.inRange(img_hsv, lower_magenta, upper_magenta)

            mask = magenta_mask

        

        elif lion_class_idx == 2:

            lower_brown = np.array([12, (85-6)*2.55, (35-5)*2.55])

            upper_brown = np.array([18, (85+16)*2.55, (35+5)*2.55])

            brown_mask = cv2.inRange(img_hsv, lower_brown, upper_brown)

            mask = brown_mask

        

        elif lion_class_idx == 3:

            lower_blue = np.array([110, 200, (67-8)*2.55])

            upper_blue = np.array([130, 255, (67+8)*2.55])

            blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

            mask = blue_mask

        

        elif lion_class_idx == 4:

            lower_green = np.array([40, 200, (67-8)*2.55])

            upper_green = np.array([70, 255, (67+8)*2.55])

            green_mask = cv2.inRange(img_hsv, lower_green, upper_green)

            mask = green_mask



        dot_places = np.where(mask!=0)

        dot_count = dot_places[0].shape[0]



        X = np.array( [dot_places[0], dot_places[1]]).T

        kmeans = KMeans(n_clusters=train_df[lion_class_str][int(img_fn)], random_state=0).fit(X)



        with open(img_fn+'.txt', 'a') as txt_file:

            for i in range(kmeans.cluster_centers_.shape[0]):

                x, y = kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1]

                txt_file.write(str(lion_class_idx)+','+str(int(x))+','+str(int(y))+'\n')