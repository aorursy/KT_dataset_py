import os

import glob

import cv2

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



from scipy import stats







%matplotlib inline

%config InlineBackend.figure_format = 'retina'
train_df = pd.read_csv('../input/landmark-retrieval-2020/train.csv')

train_df
plt.title('landmark_id distribution')

sns.distplot(train_df['landmark_id'])
sns.set()

plt.title('Training set: number of images per class(line plot)')

landmarks_fold = pd.DataFrame(train_df['landmark_id'].value_counts())

landmarks_fold.reset_index(inplace=True)

landmarks_fold.columns = ['landmark_id','count']

ax = landmarks_fold['count'].plot(logy=True, grid=True)

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images")
sns.set()

landmarks_fold_sorted = pd.DataFrame(train_df['landmark_id'].value_counts())

landmarks_fold_sorted.reset_index(inplace=True)

landmarks_fold_sorted.columns = ['landmark_id','count']

landmarks_fold_sorted = landmarks_fold_sorted.sort_values('landmark_id')

ax = landmarks_fold_sorted.plot.scatter(\

     x='landmark_id',y='count',

     title='Training set: number of images per class(statter plot)')

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images")
test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')

index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')
print( 'Query', len(test_list), ' test images in ', len(index_list), 'index images')
plt.rcParams["axes.grid"] = False

f, axarr = plt.subplots(4, 3, figsize=(24, 22))



curr_row = 0

for i in range(12):

    example = cv2.imread(test_list[i])

    example = example[:,:,::-1]

    

    col = i%4

    axarr[col, curr_row].imshow(example)

    if col == 3:

        curr_row += 1

            

#     plt.imshow(example)

#     plt.show()