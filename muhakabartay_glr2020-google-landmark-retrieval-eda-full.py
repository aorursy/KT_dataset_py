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

        #print(os.path.join(dirname, filename))

        continue



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Content

!ls /kaggle/input/landmark-retrieval-2020/

!ls /kaggle/input/landmark-retrieval-2020/train

!ls /kaggle/input/landmark-retrieval-2020/test

!ls /kaggle/input/landmark-retrieval-2020/index

print('')



# Size of data

#!du -hs /kaggle/input/landmark-retrieval-2020/train/  # 101 GB

#!du -hs /kaggle/input/landmark-retrieval-2020/test/   # 0.07 GB

#!du -hs /kaggle/input/landmark-retrieval-2020/index/  # 4.9 GB
from pathlib import Path



data_path = Path('/kaggle/input/landmark-retrieval-2020/')

train_path = data_path / 'train'

test_path = data_path / 'test'

index_path = data_path / 'index'



print('training_path =', train_path)

print('test_path     =', test_path)

print('index_path    =', index_path)
import glob

train_list = glob.glob('../input/landmark-retrieval-2020/train/*/*/*/*')

test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')

index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')

print( 'train_images :', len(train_list))

print( 'test_images  :', len(test_list))

print( 'index_images :', len(index_list))

train = pd.read_csv('../input/landmark-retrieval-2020/train.csv')

train.head()
train.shape
train.hist(bins=100)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
fig, axs = plt.subplots(figsize=(16,6))



ax = sns.distplot(train['landmark_id'])



plt.savefig('landmark_id.png', dpi=100)
fig, axs = plt.subplots(figsize=(16,6))



ax = sns.distplot(train['landmark_id'], 

                  rug=False, 

                  #rug_kws={"color": "g"}, 

                  kde_kws={"color": "k", "lw": 3, "label": "KDE"}, 

                  hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"})



plt.savefig('landmark_id_2.png', dpi=100)
train['landmark_id'].value_counts()
#train['landmark_id'].value_counts().plot(kind="bar")
## Top 20 by popularity within dataset



fig, axs = plt.subplots(figsize=(12,8))



train['landmark_id'].value_counts(sort=True, ascending=False)[:30].plot(kind='barh')



plt.title('Top 50 by popularity')

plt.xlabel('Number of images', fontsize=14)

plt.ylabel('Landmark', fontsize=14)



plt.savefig('landmark_id_top30.png', dpi=100)
image_count  = train['landmark_id'].value_counts(sort=True, ascending=False)

image_count = image_count[:30,]



plt.figure(figsize=(23,6))



sns.barplot(image_count.index, image_count.values, alpha=0.8)



plt.title('Images per landmark')

plt.ylabel('Number of images', fontsize=14)

plt.xlabel('Landmark', fontsize=14)



plt.savefig('landmark_id_top30_vs_images.png', dpi=100)



plt.show()
image_count_all  = train['landmark_id'].value_counts(sort=True, ascending=False)



print('landmark_id:')



print(image_count_all)



print()

print ('Max images for a given landmark_id: {}'.format(image_count_all.max()))

print ('Min images for a given landmark_id: {}'.format(image_count_all.min()))



print()

print('Number of unique landmark IDs: {}'.format(len(image_count_all))) # len(image_count_all.index.values)
df_image_count_all = pd.DataFrame(image_count_all.reset_index().values, columns=['landmark_id', 'Number' ])

df_image_count_all_ind = df_image_count_all.sort_index(axis = 0, ascending=True)

df_image_count_all_ind
df_image_count_all[df_image_count_all['Number']<3].count()
df_image_count_all[df_image_count_all['Number']<11].count()
df_image_count_all[df_image_count_all['Number']<1001].count()
df_image_count_all[df_image_count_all['Number']>1001].count()
num_list = list(df_image_count_all['Number'])



fig, axs = plt.subplots(figsize=(23,6))

plt.hist(num_list, density=False, bins=6272, color='green', alpha=0.5)  



axs.set_xscale('log')



plt.ylabel('Counts') # Probability if scaled

plt.xlabel('Number of images per Landmark ID')



plt.savefig('images_per_landmark_id.png', dpi=100)



plt.show()
num_list = list(df_image_count_all['Number'])



fig, axs = plt.subplots(figsize=(23,6))

plt.hist(num_list, density=False, bins=6272, color='green', alpha=0.5)  



axs.set_xscale('log')

axs.set_yscale('log')



plt.ylabel('Counts') # Probability if scaled

plt.xlabel('Number of images per Landmark ID')



plt.savefig('images_per_landmark_id_2.png', dpi=100)



plt.show()
#plt.figure(figsize=(23,6))



fig, axs = plt.subplots(2,1, figsize=(23,12))



# Normal version

sns.lineplot(x=df_image_count_all_ind['landmark_id'], y=df_image_count_all_ind['Number'], data=df_image_count_all_ind, color='green', alpha=0.7, ax=axs[0])

# Log version

plott = sns.lineplot(x=df_image_count_all_ind['landmark_id'], y=df_image_count_all_ind['Number'], data=df_image_count_all_ind, color='green', alpha=0.5, ax=axs[1])

plott.set(yscale="log")



plt.savefig('landmark_id_distribution_nonlog_log.png', dpi=100)



plt.show()
import cv2



plt.rcParams["axes.grid"] = False

fig, axs = plt.subplots(4, 4, figsize=(23, 23))



curr_row = 0



for i in range(0,16):

    example = cv2.imread(test_list[i])

    example = example[:,:,::-1]

    

    col = i%4

    

    ## Add title information

    axs[i//4, i%4].set_title('{}/{}/{}/{}'.format(

        test_list[i].split('/')[-4],

        test_list[i].split('/')[-3],

        test_list[i].split('/')[-2],

        test_list[i].split('/')[-1]))

    

    ## Plot n x m images

    axs[col, curr_row].imshow(example)

    

    if col == 3:

        curr_row += 1

    

plt.savefig('some_data_images.png', dpi=100)



plt.show()