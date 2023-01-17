from math import ceil

from pathlib import Path



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')  

path_train_image_dir = Path('/kaggle/input/global-wheat-detection/train/')
df_train.head(5)
df_train['bbox'][0]
type(df_train['bbox'][0])
bboxes = np.stack(df_train['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x_min', 'y_min', 'width', 'height']):

    df_train[column] = bboxes[:,i]

    

df_train["x_max"] = df_train.apply(lambda col: col.x_min + col.width, axis=1)

df_train["y_max"] = df_train.apply(lambda col: col.y_min + col.height, axis = 1)

df_train.drop(columns=['bbox'], inplace=True)
df_train.head()
display(df_train[df_train["x_max"] > 1024])

display(df_train[df_train["y_max"] > 1024])

display(df_train[df_train["x_min"] < 0])

display(df_train[df_train["y_min"] < 0])
list_image_filepath = list(path_train_image_dir.glob('*.jpg'))

len(list_image_filepath)
list_image_conf = list()

for idx, image_filepath in enumerate(list_image_filepath): 

    image = cv2.imread(str(image_filepath))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    width = image.shape[0]

    height = image.shape[1]

    area = width * height

    list_image_conf.append(pd.Series(

        [image_filepath.stem, width, height, area], 

        index=['filename', 'width', 'height', 'area']

    ))

    

df_image_conf = pd.concat(list_image_conf, axis=1).T

    
df_image_conf.head(5)
df_image_conf['width'].value_counts()
df_image_conf['height'].value_counts()
df_image_conf['area'].value_counts()
df_train[['x_min', 'x_max', 'y_min', 'y_max']].max()
len(df_train)
df_train.iloc[31785]
df_train = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv') 

bboxes = np.stack(df_train['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
bboxes = np.round(bboxes)

display(bboxes[:5])
for i, column in enumerate(['x_min', 'y_min', 'width', 'height']):

    df_train[column] = bboxes[:,i]

    

df_train[['x_min', 'y_min', 'width', 'height']] = df_train[['x_min', 'y_min', 'width', 'height']].astype(int)

df_train.dtypes
df_train["x_max"] = df_train.apply(lambda col: col.x_min + col.width, axis=1)

df_train["y_max"] = df_train.apply(lambda col: col.y_min + col.height, axis = 1)

df_train.drop(columns=['bbox'], inplace=True)



display(df_train.head(5))

display(df_train.dtypes)
display(df_train[df_train["x_max"] > 1024])

display(df_train[df_train["y_max"] > 1024])

display(df_train[df_train["x_min"] < 0])

display(df_train[df_train["y_min"] < 0])
df_train['class'] = 1
df_train['image_filename'] = df_train['image_id'].apply(lambda x: f'{x}.jpg')
def plot_random_images(image_folder_path, df_image_annotation, num = 12):

    img_dict = {}

    list_image_name = df_image_annotation['image_filename'].unique().tolist()



    # randomly choose 12 image.

    img_files_list = np.random.choice(list_image_name, num)



    img_matrix_list = []

    for img_file in img_files_list:

        image_file_path = image_folder_path/img_file

        img = cv2.imread(str(image_file_path))

        img_matrix_list.append(img)



    fig, axes = plt.subplots(ceil(num / 4), 4, figsize=(12, 9))

    for idx, image in enumerate(img_matrix_list):

        idx_row = idx // 4

        idx_col = idx % 4

        axes[idx_row][idx_col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        axes[idx_row][idx_col].set_title(img_files_list[idx])

        

    plt.tight_layout()

    plt.show()

    plt.close()
plot_random_images(path_train_image_dir, df_train, num=12)
def draw_rect(image, arr_bboxes, color=(0, 0, 255)):

    for idx, bbox in enumerate(arr_bboxes): 

        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

        

    return image



def plot_random_images_bbox(image_folder_path, df_image_annotation, num = 12):

    img_dict = {}

    for idx in range(len(df_image_annotation)):

        image_name = df_image_annotation['image_filename'][idx]

        x_min = df_image_annotation['x_min'][idx]

        y_min = df_image_annotation['y_min'][idx]

        x_max = df_image_annotation['x_max'][idx]

        y_max = df_image_annotation['y_max'][idx]



        if image_name not in img_dict:

            img_dict[image_name] = list()

        img_dict[image_name].append([x_min, y_min, x_max, y_max])



    # randomly choose 12 image.

    img_files_list = np.random.choice(list(img_dict.keys()), num)



    bbox_list = []

    img_matrix_list = []

    

    for img_file in img_files_list:

        image_file_path = image_folder_path/img_file

        img = cv2.imread(str(image_file_path))

        bbox_list.append(img_dict[img_file])

        img_matrix_list.append(img)



    final_bbox_list = []

    for bboxes, img in zip(bbox_list, img_matrix_list):

        final_bbox_array = np.array([])

        for bbox in bboxes:

            bbox = np.array(bbox).reshape(1,4)

            final_bbox_array = np.append(final_bbox_array, bbox)

        final_bbox_array = final_bbox_array.reshape(-1,4).astype(int)

        image_with_bboxes = draw_rect(img.copy(), final_bbox_array.copy(), color=(0, 0, 255))

        final_bbox_list.append(image_with_bboxes)



    fig, axes = plt.subplots(ceil(num / 4), 4, figsize=(12, 9))

    for idx, image in enumerate(final_bbox_list):

        idx_row = idx // 4

        idx_col = idx % 4

        axes[idx_row][idx_col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        axes[idx_row][idx_col].set_title(img_files_list[idx])

        

    plt.tight_layout()

    plt.show()

    plt.close()
plot_random_images_bbox(path_train_image_dir, df_train, num=12)