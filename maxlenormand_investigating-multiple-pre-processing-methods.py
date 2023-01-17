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

print(f"Shape of df_0: {df_0.shape} (took {time.time() - start_time}sec to load)")

current_time = time.time()



df_1 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_1.parquet')

print(f"Shape of df_1: {df_1.shape} (took {time.time() - current_time}sec to load)")

current_time = time.time()



df_2 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_2.parquet')

print(f"Shape of df_2: {df_2.shape} (took {time.time() - current_time}sec to load)")

current_time = time.time()



df_3 = pd.read_parquet('/kaggle/input/bengaliai-cv19/train_image_data_3.parquet')



print(f"It took: {time.time() - start_time} to load all 4 datasets")
HEIGHT = 137

WIDTH = 236

CROP_SIZE = 75
original_img_size = HEIGHT * WIDTH

cropped_img_size = CROP_SIZE * CROP_SIZE



print(f"Original shape of images: {original_img_size}\nCropped & resized shape of images: {cropped_img_size}")

print(f"Reduction fatio: {np.round(original_img_size/cropped_img_size, 3)}")
resized = df_0.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
DEFAULT_PADDING_SIZE = int((CROP_SIZE)*.05)

print(f'Default padding size: {DEFAULT_PADDING_SIZE}')



def crop_and_resize_with_interpolation_and_correction_images(df,

                                                             resized_df, 

                                                             resize_size = CROP_SIZE,

                                                             padding_size = DEFAULT_PADDING_SIZE, 

                                                             remove_noise = False,

                                                             blur=False, 

                                                             Laplace_filter=False):

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

        # Padding:

        padded_roi = cv2.copyMakeBorder(roi, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=[255])



        resized_roi = cv2.resize(padded_roi, (resize_size, resize_size), interpolation = cv2.INTER_AREA)

        

        #Noise Removing

        if remove_noise:

            resized_roi=cv2.fastNlMeansDenoising(resized_roi)

        

        #Gaussian Blur

        if blur:

            gaussian = cv2.GaussianBlur(resized_roi, (9,9), 10.0)

            resized_roi = cv2.addWeighted(resized_roi, 1.5, gaussian, -0.5, 0, resized_roi)

            

        #Laplacian Filter

        if Laplace_filter:

            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) #filter

            resized_roi = cv2.filter2D(resized_roi, -1, kernel)

        

        cropped_imgs[df.image_id[img_id]] = resized_roi.reshape(-1)

        

    resized = pd.DataFrame(cropped_imgs).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized
IMGS_TO_TEST = 5
test_df_no_blur_no_filter = crop_and_resize_with_interpolation_and_correction_images(df_0.head(IMGS_TO_TEST), resized, CROP_SIZE, 

                                                                                     blur=False, 

                                                                                     Laplace_filter=False)

resized_test = test_df_no_blur_no_filter.iloc[:, 1:].values.reshape(-1, CROP_SIZE, CROP_SIZE)







test_df_with_blur_no_filter = crop_and_resize_with_interpolation_and_correction_images(df_0.head(IMGS_TO_TEST), resized, CROP_SIZE, 

                                                                                       blur=True, 

                                                                                       Laplace_filter=False)

resized_test_blur = test_df_with_blur_no_filter.iloc[:, 1:].values.reshape(-1, CROP_SIZE, CROP_SIZE)







test_df_with_blur_with_filter = crop_and_resize_with_interpolation_and_correction_images(df_0.head(IMGS_TO_TEST), resized, CROP_SIZE, 

                                                                                         blur=True, 

                                                                                         Laplace_filter=True)

resized_test_blur_filter = test_df_with_blur_with_filter.iloc[:, 1:].values.reshape(-1, CROP_SIZE, CROP_SIZE)







test_df_with_blur_with_filter_with_noise_removal = crop_and_resize_with_interpolation_and_correction_images(df_0.head(IMGS_TO_TEST), resized, CROP_SIZE,

                                                                                                            remove_noise=True, 

                                                                                                            blur=True,

                                                                                                            Laplace_filter=True)

resized_test_blur_filter_noise_removed = test_df_with_blur_with_filter_with_noise_removal.iloc[:, 1:].values.reshape(-1, CROP_SIZE, CROP_SIZE)



for img in range(5):

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(30, 15))

    ax0.imshow(resized[img], cmap='Greys')

    ax0.set_title('Original image')

    ax1.imshow(resized_test[img], cmap='Greys')

    ax1.set_title('Resized & cropped image')

    ax2.imshow(resized_test_blur[img], cmap='Greys')

    ax2.set_title('Resized, cropped and blurred image')

    ax3.imshow(resized_test_blur_filter[img], cmap='Greys')

    ax3.set_title('Resized & cropped, blurred, filter')

    ax4.imshow(resized_test_blur_filter_noise_removed[img], cmap='Greys')

    ax4.set_title('Resized & cropped, blurred, filter & noise removed')

    plt.show()
# start = time.time()



# # dataset 1

# resized_0 = df_0.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

# cropped_df_0 = crop_and_resize_with_interpolation_and_correction_images(df_0, resized_0, CROP_SIZE,remove_noise=True,blur=True,Laplace_filter=True)

# cropped_df_0.to_feather("train_data_0.feather")

# del resized_0

# del cropped_df_0

# print(f"Saved cropped & resized df_0 to feather in {time.time() - start}sec")

# current = time.time()



# # dataset 1

# resized_1 = df_1.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

# cropped_df_1 = crop_and_resize_with_interpolation_and_correction_images(df_1, resized_1, CROP_SIZE,remove_noise=True,blur=True,Laplace_filter=True)

# cropped_df_1.to_feather("train_data_1.feather")

# del resized_1

# del cropped_df_1

# print(f"Saved cropped & resized df_1 to feather in {time.time() - start}sec")

# current = time.time()



# # dataset 2

# resized_2 = df_2.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

# cropped_df_2 = crop_and_resize_with_interpolation_and_correction_images(df_2, resized_2, CROP_SIZE,remove_noise=True,blur=True,Laplace_filter=True)

# cropped_df_2.to_feather("train_data_2.feather")

# del resized_2

# del cropped_df_2

# print(f"Saved cropped & resized df_2 to feather in {time.time() - current}sec")

# current = time.time()



# # dataset 3

# resized_3 = df_3.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

# cropped_df_3 = crop_and_resize_with_interpolation_and_correction_images(df_3, resized_3, CROP_SIZE,remove_noise=True,blur=True,Laplace_filter=True)

# cropped_df_3.to_feather("train_data_3.feather")

# del resized_3

# del cropped_df_3

# print(f"Saved cropped & resized df_3 to feather in {time.time() - current}sec")