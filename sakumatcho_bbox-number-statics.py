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
list_jpg = list()

for dirname, _, filenames in os.walk('/kaggle/input'):

    if dirname != '/kaggle/input/global-wheat-detection/train': 

        continue 

        

    for filename in filenames:

        if filename[-4:] != '.jpg': 

            print(filename)

            continue 

            

        list_jpg.append(filename[:-4])



df_jpg_filename = pd.DataFrame(list_jpg, columns=['image_id'])





df_num_bb = df_train['image_id'].value_counts().reset_index()

df_num_bb.columns = ['image_id', 'count']



df_num_bb_per_image = pd.merge(df_jpg_filename, df_num_bb, how='left', on='image_id').fillna(0)



display(df_num_bb_per_image)
df_num_bb_distribution = df_num_bb_per_image['count'].value_counts().sort_index().reset_index()

df_num_bb_distribution.columns = ['number', 'count']

display(df_num_bb_distribution)
fig = plt.figure(figsize=(20, 5))



plt.bar(df_num_bb_distribution['number'], df_num_bb_distribution['count'])



plt.title('Bounding-box distribution')

plt.xlabel('number per image')

plt.ylabel('counts')



plt.show()

plt.close()
min_num_box = df_num_bb_distribution['number'].min()

few_num_box = df_num_bb_distribution['number'][1]

max_num_box = df_num_bb_distribution['number'].max()

ave_num_box = np.round(df_num_bb_distribution['number'].mean(), 0)

std_num_box = np.round(df_num_bb_distribution['number'].std(), 0)



num_popular = df_num_bb_distribution['count'].max()

popular_num_box = df_num_bb_distribution[df_num_bb_distribution['count'] == num_popular]['number'].values.tolist()



print(f'min_num_box: {min_num_box}')

print(f'few_num_box: {few_num_box}')

print(f'max_num_box: {max_num_box}')

print(f'ave_num_box: {ave_num_box}')

print(f'std_num_box: {std_num_box}')

print(f'popular_num_box: {popular_num_box}')
query_min = df_num_bb_per_image['count'] == min_num_box

sr_image_id_has_min_num_box = df_num_bb_per_image[query_min]['image_id']



query_few = df_num_bb_per_image['count'] == few_num_box

sr_image_id_has_few_num_box = df_num_bb_per_image[query_few]['image_id']



query_max = df_num_bb_per_image['count'] == max_num_box

sr_image_id_has_max_num_box = df_num_bb_per_image[query_max]['image_id']



query_ave = df_num_bb_per_image['count'] == ave_num_box

sr_image_id_has_ave_num_box = df_num_bb_per_image[query_ave]['image_id']



list_image_id_popular = list()

for pop_num_box in popular_num_box:

    query_popular = df_num_bb_per_image['count'] == pop_num_box

    list_image_id_popular.extend(df_num_bb_per_image[query_popular]['image_id'].values.tolist())



print(f'image_id_has_min_num_box:\n {sr_image_id_has_min_num_box.values.tolist()}')

print(f'image_id_has_few_num_box:\n {sr_image_id_has_few_num_box.values.tolist()}')

print(f'image_id_has_max_num_box:\n {sr_image_id_has_max_num_box.values.tolist()}')

print(f'image_id_has_ave_num_box:\n {sr_image_id_has_ave_num_box.values.tolist()}')

print(f'image_id_has_popular_num_box:\n {list_image_id_popular}')
bboxes = np.stack(df_train['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=','))).astype(int)
for i, column in enumerate(['x_min', 'y_min', 'width', 'height']):

    df_train[column] = bboxes[:,i]

    

df_train["x_max"] = df_train.apply(lambda col: col.x_min + col.width, axis=1)

df_train["y_max"] = df_train.apply(lambda col: col.y_min + col.height, axis = 1)

df_train.drop(columns=['bbox'], inplace=True)
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
ext_image_id = np.random.choice(sr_image_id_has_min_num_box, 8, replace=False)

print(ext_image_id)



plot_images(df_train, ext_image_id)
ext_image_id = np.random.choice(sr_image_id_has_few_num_box, 8, replace=False)

print(ext_image_id)



plot_images(df_train, ext_image_id)
plot_images(df_train, sr_image_id_has_max_num_box.values.tolist())
ext_image_id = np.random.choice(sr_image_id_has_ave_num_box, 8, replace=False)

print(ext_image_id)



plot_images(df_train, ext_image_id)
ext_image_id = np.random.choice(list_image_id_popular, 8, replace=False)

print(ext_image_id)



plot_images(df_train, ext_image_id)