import numpy as np

import pandas as pd

import cv2



import time

import os

from tqdm import tqdm



import matplotlib.pyplot as plt

import seaborn as sns
start_time = time.time()

df_0 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_0.parquet')

current_time = time.time()

print(f"Shape: {df_0.shape} (took {time.time() - start_time}sec to load)")



df_1 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

current_time = time.time()

print(f"Shape: {df_1.shape} (took {time.time() - current_time}sec to load)")



df_2 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_2.parquet')

current_time = time.time()

print(f"Shape: {df_2.shape} (took {time.time() - current_time}sec to load)")



df_3 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_3.parquet')



print(f"It took: {time.time() - start_time} to load all 4 datasets")
HEIGHT = 137

WIDTH = 236



CROP_SIZE = 100
original_img_size = HEIGHT * WIDTH



cropped_img_size = CROP_SIZE * CROP_SIZE



print(f"Original shape of images: {original_img_size}\nCropped & resized shape of images: {cropped_img_size}")

print(f"Reduction fatio: {np.round(original_img_size/cropped_img_size, 3)}")
def crop_and_resize_images(df, resized_df, resize_size = CROP_SIZE):

    cropped_imgs = {}

    for img_id in tqdm(range(df.shape[0])):

        img = resized_df[img_id]

        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

        

        idx = 0 

        ls_xmin = []

        ls_ymin = []

        ls_xmax = []

        ls_ymax = []

        for cnt in contours:

            idx += 1

            x,y,w,h = cv2.boundingRect(cnt)

            ls_xmin.append(x)

            ls_ymin.append(y)

            ls_xmax.append(x + w)

            ls_ymax.append(y + h)

        xmin = min(ls_xmin)

        ymin = min(ls_ymin)

        xmax = max(ls_xmax)

        ymax = max(ls_ymax)



        roi = img[ymin:ymax,xmin:xmax]

        resized_roi = cv2.resize(roi, (resize_size, resize_size))

        cropped_imgs[df.image_id[img_id]] = resized_roi.reshape(-1)

        

    resized = pd.DataFrame(cropped_imgs).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized #out_df
resized = df_0.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
cropped_df = crop_and_resize_images(df_0, resized, CROP_SIZE)
cropped_df.head()
cropped_df.to_feather("train_data_0.feather")
sample_df = pd.read_feather("train_data_0.feather")
resized_sample = sample_df.iloc[:, 1:].values.reshape(-1, CROP_SIZE, CROP_SIZE)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 15))

ax0.imshow(resized[329], cmap='Greys')

ax0.set_title('Original image')

ax1.imshow(resized_sample[329], cmap='Greys')

ax1.set_title('Resized & cropped image')

plt.show()
del resized

del cropped_df
start = time.time()



# dataset 1

resized_1 = df_1.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

cropped_df_1 = crop_and_resize_images(df_1, resized_1, CROP_SIZE)

cropped_df_1.to_feather("train_data_1.feather")

del resized_1

del cropped_df_1

print(f"Saved cropped & resized df_1 to feather in {time.time() - start}sec")

current = time.time()



# dataset 2

resized_2 = df_2.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

cropped_df_2 = crop_and_resize_images(df_2, resized_2, CROP_SIZE)

cropped_df_2.to_feather("train_data_2.feather")

del resized_2

del cropped_df_2

print(f"Saved cropped & resized df_2 to feather in {time.time() - current}sec")

current = time.time()



# dataset 3

resized_3 = df_3.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

cropped_df_3 = crop_and_resize_images(df_3, resized_3, CROP_SIZE)

cropped_df_3.to_feather("train_data_3.feather")

del resized_3

del cropped_df_3

print(f"Saved cropped & resized df_3 to feather in {time.time() - current}sec")