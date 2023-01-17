import matplotlib.pyplot as plt

from matplotlib import patches

import numpy as np

import pandas as pd

import cv2

import os

import ast

from scipy.stats import norm, kstest, describe



IMG_PATH = "/kaggle/input/global-wheat-detection/train"

SEED = 42





train_df = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")

np.random.seed(SEED)
indexes = [np.random.randint(0, len(train_df)) for _ in range(8)]



fig, ax = plt.subplots(2, 4, figsize=(20,10))



for i, idx in enumerate(indexes):

    idx_row = int(i / 4)

    idx_col = i % 4

    

    filename = train_df.iloc[idx, 0]

    

    ax[idx_row][idx_col].imshow(

        cv2.imread(os.path.join(IMG_PATH, filename+".jpg"))

    )

    

    for _, row in train_df[train_df["image_id"] == filename].iterrows():

        bbox = ast.literal_eval(row["bbox"])

        rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')

        

        ax[idx_row][idx_col].add_patch(rect)
val_count = train_df["image_id"].value_counts()

np_val_count = val_count.to_numpy() # we obtain here an array that maps, for each image, how many bbox they have.

#we now need to count how often a number of bbox/image is 
empirical_mu = np.mean(np_val_count)

empirical_sigma = np.std(np_val_count)



unique_elements, count_elements = np.unique(np_val_count, return_counts=True)

x_all = np.arange(-10, 120, 0.001) # entire range of x, both in and out of spec

y2 = norm.pdf(x_all, empirical_mu, empirical_sigma)*len(np_val_count)



fig = plt.figure(figsize=(20, 7))

plt.xlabel("Count of the number of bbox")

plt.ylabel("Frequency of apparition of bbox count")



line = plt.plot(unique_elements, count_elements)

line[0].set_label(f"Frequency of bbox count per image (mean: {np.round(empirical_mu,2)}, std:{np.round(empirical_sigma, 2)})")

line = plt.plot(x_all, y2)

line[0].set_label(f"Normal approximation of the BBox distribution")



plt.legend()

plt.show()
kstest(count_elements, 'norm')
max_filenames = val_count.index[0:4]

max_bboxes = val_count[0:4]



min_filenames = val_count.index[-5:-1]

min_bboxes = val_count[-5:-1]



print(f"The file with the biggest amount of bbox is '{max_filenames[0]}' with {max_bboxes[0]} bboxes \nwhile the file with the least amount of bbox is '{min_filenames[0]}' with {min_bboxes[0]} bbox")
fig_max, ax_max = plt.subplots(1, 4, figsize=(20,5))

fig_min, ax_min = plt.subplots(1, 4, figsize=(20,5))



fig_max.suptitle("Plots of images with the most BBoxes")

fig_min.suptitle("Plots of images with the least BBoxes")



for i in range(len(max_filenames)):

    

    ax_max[i].imshow(

        cv2.imread(os.path.join(IMG_PATH, max_filenames[i]+".jpg"))

    ) # plotting image with the more bboxes

    

    ax_max[i].set_title(f"'{max_filenames[i]+'.jpg'}' with {max_bboxes[i]} bboxes")

    

    for _, row in train_df[train_df["image_id"] == max_filenames[i]].iterrows():

        bbox = ast.literal_eval(row["bbox"])

        rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')

        

        ax_max[i].add_patch(rect)

    

    ax_min[i].imshow(

        cv2.imread(os.path.join(IMG_PATH, min_filenames[i]+".jpg"))

    ) # plotting image with the more bboxes

        

    ax_min[i].set_title(f"'{min_filenames[i]+'.jpg'}' with {min_bboxes[i]} bbox")

    

    for _, row in train_df[train_df["image_id"] == min_filenames[i]].iterrows():

        bbox = ast.literal_eval(row["bbox"])

        rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')

        

        ax_min[i].add_patch(rect)
def list_to_area(l):

    return l[2]*l[3]

    

areas = train_df["bbox"].apply(lambda l: list_to_area(ast.literal_eval(l))).to_numpy()

describe(areas)
bboxes = train_df["bbox"].apply(lambda l: ast.literal_eval(l)).to_numpy()

bboxes
widths = [bbox[2] for bbox in bboxes]

heights = [bbox[3] for bbox in bboxes]
plt.figure(figsize=(15,10))

plt.xlabel("Width of a BBox")

plt.ylabel("Height of a BBox")

plt.title("Scatter plot of WIdth to Height ration of BBoxes")

plt.scatter(widths, heights)

plt.show()
outliers = [(idx, bbox) for idx, bbox in enumerate(bboxes) if (bbox[2] > 600 or bbox[3] > 600)]

outliers
outliers_idx, outliers_bboxes = list(zip(*outliers))
outliers_idx, outliers_bboxes = list(outliers_idx), list(outliers_bboxes)
fig, axs = plt.subplots(2,3, figsize=(15,10))



for i, idx in enumerate(outliers_idx):

    

    x = int(i/3)

    y = i % 3

    

    bbox = outliers_bboxes[i]



    filename = str(train_df.iloc[idx, 0])+".jpg"

    

    axs[x][y].imshow(

        cv2.imread(

            os.path.join(IMG_PATH,filename)

        )

    )

    

    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')

    axs[x][y].add_patch(rect)