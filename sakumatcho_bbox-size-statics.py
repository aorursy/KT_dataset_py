import os

from math import log2, ceil

from pathlib import Path



import pandas as pd 

from matplotlib import pyplot as plt 

import numpy as np

import cv2

%matplotlib inline
path_dir_project = Path('/kaggle/input/global-wheat-detection')
df_train = pd.read_csv(path_dir_project/'train.csv')
bboxes = np.stack(df_train['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=','))).astype(int)
display(bboxes[:5])
for i, column in enumerate(['x_min', 'y_min', 'width', 'height']):

    df_train[column] = bboxes[:,i]

    

df_train["x_max"] = df_train.apply(lambda col: col.x_min + col.width, axis=1)

df_train["y_max"] = df_train.apply(lambda col: col.y_min + col.height, axis = 1)

df_train.drop(columns=['bbox'], inplace=True)
df_train[['x_min', 'y_min', 'width', 'height', 'x_max', 'y_max']].head(5)
df_train['area'] = df_train['width'] * df_train['height']
df_train[['x_min', 'y_min', 'width', 'height', 'x_max', 'y_max', 'area']].head(5)
len(df_train)
df_train['area'].describe()
sr_area = df_train['area'].value_counts()

display(sr_area)
fig = plt.figure(figsize=(20, 5))



plt.bar(sr_area.sort_index().index, sr_area.sort_index().values)



plt.title('Bounding-box distribution')

plt.xlabel('Area [dots]')

plt.ylabel('counts')



plt.show()

plt.close()
def calc_starges(num_data):

    return int(np.round(log2(num_data) + 1, 0))
fig = plt.figure(figsize=(20, 5))



plt.hist(df_train['area'], bins=calc_starges(len(df_train)))



plt.title('Bounding-box distribution')

plt.xlabel('Area [dots]')

plt.ylabel('counts')



plt.show()

plt.close()
fig = plt.figure(figsize=(20, 5))



plt.hist(df_train['area'], bins=calc_starges(len(df_train)))



plt.title('Bounding-box distribution')

plt.xlabel('Area [dots]')

plt.ylabel('counts')

plt.yscale('log')



plt.show()

plt.close()
fig = plt.figure(figsize=(20, 5))



plt.bar(sr_area.sort_index().index, sr_area.sort_index().values)



plt.title('Bounding-box distribution')

plt.xlabel('Area [dots]')

plt.ylabel('counts')

plt.xscale('log')

plt.yscale('log')

plt.show()

plt.close()
query_max_area = df_train['area'] == df_train['area'].max()

image_id_max_area = df_train[query_max_area]['image_id']

print(image_id_max_area)
fig_filename = f'{image_id_max_area.values[0]}.jpg'

filepath = path_dir_project/f'train/{fig_filename}'
image = cv2.imread(str(filepath))
fig = plt.figure(figsize=(8, 8))



plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.show()

plt.close()
query_target = df_train['image_id'] == image_id_max_area.values[0]

bboxes_target = df_train[query_target]
len(bboxes_target)
bboxes_target.head(5)
image_bboxes = image.copy()

for idx in range(len(bboxes_target)): 

    bbox = bboxes_target.iloc[idx][['x_min', 'y_min', 'x_max', 'y_max']].values

    image_bboxes = cv2.rectangle(image_bboxes, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)

    

fig = plt.figure(figsize=(8, 8))



plt.imshow(cv2.cvtColor(image_bboxes, cv2.COLOR_BGR2RGB))

plt.show()

plt.close()
query_large = df_train['area'] >= 100000 

df_train_large = df_train[query_large]

display(df_train_large)
len(df_train[query_large]['image_id'].unique())
image_ids_large_area = df_train[query_large]['image_id'].unique()
def draw_bbox(image_id, df_train):

    fig_filename = f'{image_id}.jpg'

    filepath = path_dir_project/f'train/{fig_filename}'



    image = cv2.imread(str(filepath))



    query_target = df_train['image_id'] == image_id

    bboxes_target = df_train[query_target][['x_min', 'y_min', 'x_max', 'y_max']]



    for idx in range(len(bboxes_target)): 

        bbox = bboxes_target.iloc[idx].values

        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 5)

        

    return image



def plot_images(df_train, image_ids):

    num_images = len(image_ids)

    num_row = ceil(num_images / 2)

    if num_images >= 2:

        fig, axes = plt.subplots(num_row, 2, figsize=(20, 10 * num_row))

        for idx_img, image_id in enumerate(image_ids):

            image_bboxes = draw_bbox(image_id, df_train)

            axes[idx_img // 2][idx_img % 2].imshow(cv2.cvtColor(image_bboxes, cv2.COLOR_BGR2RGB))

            axes[idx_img // 2][idx_img % 2].set_title(image_id)

            

    else: 

        image_id = image_ids[0]

        fig = plt.figure(figsize=(7, 6))



        image_bboxes = draw_bbox(image_id, df_train)

        plt.imshow(cv2.cvtColor(image_bboxes, cv2.COLOR_BGR2RGB))

        plt.title(image_id)        



    plt.show()

    plt.close()    
plot_images(df_train, image_ids_large_area)
images_with_anormally_large_bbox = [

    '41c0123cc', 

    'a1321ca95', 

    '2cc75e9f5', 

    '42e6efaaa', 

    '409a8490c', 

    'd7a02151d', 

    'd067ac2b1',

    'd60e832a5'

]
query_min = df_train['area'] == df_train['area'].min()

df_train_min = df_train[query_min]

display(df_train_min)



image_ids_min_area = df_train_min['image_id'].unique()

display(len(image_ids_min_area))
plot_images(df_train, image_ids_min_area)
query_smaller = df_train['area'] <= 100

df_train_smaller = df_train[query_smaller]

display(df_train_smaller)



image_ids_smaller_area = df_train_smaller['image_id'].unique()

display(len(image_ids_smaller_area))
plot_images(df_train, image_ids_smaller_area)
images_with_anormally_small_bbox = [

    '809d816dd', 

    '8e1543437', 

    'c3fa071f6', 

    '666e0a853', 

    'b2f2a5dfd', 

    '6284044ed', 

    '0a97d54ae', 

    'ad256655b', 

    '088d3df51', 

    '233cb8750', 

    '6a8522f06', 

    'ca5e51e59', 

    '5e182e37e', 

    'ce3999eb9', 

    'd0912a485', 

    'f24698e88', 

    'bf2027ed3', 

    '4ffeedce2', 

    '229716799'

]
query_small = (df_train['area'] > 100) & (400 >= df_train['area'])

df_train_small = df_train[query_small]

display(df_train_small)



image_ids_small_area = df_train_small['image_id'].unique()

display(len(image_ids_small_area))
ids_random = np.random.choice(image_ids_small_area, 10)

plot_images(df_train, ids_random)
query_small_width_height = (df_train['width'] <= 50) & (df_train['height'] <= 50)

df_train_query_small_width_height = df_train[query_small_width_height]

display(df_train_query_small_width_height)



image_ids_small_width_height = df_train_query_small_width_height['image_id'].unique()

display(len(image_ids_small_width_height))
ids_random_small_width_height = np.random.choice(image_ids_small_width_height, 10)

plot_images(df_train, ids_random_small_width_height)
query_small_width_height_or = (df_train['width'] <= 50) | (df_train['height'] <= 50)

df_train_query_small_width_height_or = df_train[query_small_width_height_or]

display(df_train_query_small_width_height_or)



image_ids_small_width_height_or = df_train_query_small_width_height_or['image_id'].unique()

display(len(image_ids_small_width_height_or))
ids_random_small_width_height_or = np.random.choice(image_ids_small_width_height_or, 10)

plot_images(df_train, ids_random_small_width_height_or)