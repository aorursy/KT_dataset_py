import os

import gc

import numpy as np

import pandas as pd

import skimage.io

import cv2
df_train = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv')

df_train.head()
result_dicts = []

for i, row in df_train.iterrows():

    if i%1000 == 0:

        for i in range(10):

            gc.collect()  # to be really effective need to run it several times... 

    

    # new columns in csv with default values

    row["size"] = -1

    for j in range(6):

        row[f"gleason_{j}_size"] = -1



    image_id = row["image_id"]

    mimg_mask = skimage.io.MultiImage("/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/" + image_id + "_mask.tiff")

    if len(mimg_mask) < 1:

        result_dicts += [row]

        continue

              

    # for this rough estiamte take only smallest zoom

    mask = mimg_mask[2][...,0]

    # be conservative and dilate pixels

    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((21,21),np.uint8))

    row["size"] = mask.size

    for j in range(6):

        row[f"size_gleason_{j}"] = np.sum(mask==j)

    result_dicts += [row]

    

    del mimg_mask

    del mask



df_train = pd.DataFrame(result_dicts)

df_train.head()
df_train["size_tissue"] = df_train[[f"size_gleason_{i}" for i in range(6)]].sum(axis=1)

for i in range(6):

    df_train[f"percent_gleason_{i}"] = df_train[f"size_gleason_{i}"] / df_train["size_tissue"] * 100
df_train.describe()
def test_mask(df, query, comment):

    print(f"{comment}: {len(df.query(query))}")

    if "good_mask" not in df:

        df.loc[:, "good_mask"] = True

    df_good = df.query(f"not ({query})")

    df_bad = df.query(query)

    df_bad.loc[:, "good_mask"] = False

    return pd.concat([df_good, df_bad], sort=False)
# no data

df_train = test_mask(df_train, "size < 1", "no mask at all")
# gleason-score (most common + second most common pattern) | ISUP Grade

# 3+3 | 1

# 3+4 | 2

# 4+3 | 3

# 4+4 | 4

# 3+5 | 4

# 5+3 | 4

# 4+5 | 5

# 5+4 | 5

# 5+5 | 5
# isup 1

df_train = test_mask(df_train, "(percent_gleason_4 > 5 or percent_gleason_5 > 5) and isup_grade == 1", "gleason pattern too large for isup 1")

df_train = test_mask(df_train, "percent_gleason_3 < 5 and isup_grade == 1", "gleason pattern too small for isup 1")
# isup 2

df_train = test_mask(df_train, "(percent_gleason_5 > 5) and isup_grade == 2", "gleason pattern too large for isup 2")

df_train = test_mask(df_train, "(percent_gleason_3 < 5 or percent_gleason_4 < 5) and isup_grade == 2", "gleason pattern too small for isup 2")
# isup 3

df_train = test_mask(df_train, "(percent_gleason_5 > 5) and isup_grade == 3", "gleason pattern too large for isup 3")

df_train = test_mask(df_train, "(percent_gleason_3 < 5 and percent_gleason_4 < 5) and isup_grade == 3", "gleason pattern too small for isup 3")
# isup 4

df_train = test_mask(df_train, "(percent_gleason_5 > 10) and isup_grade == 4", "gleason pattern too large for isup 4")

df_train = test_mask(df_train, "(percent_gleason_4 < 10 or (percent_gleason_3 < 5 and percent_gleason_5 < 5)) and isup_grade == 4", "gleason pattern too small for isup 4")
# isup 5

df_train = test_mask(df_train, "(percent_gleason_5 < 5) and isup_grade == 5", "gleason pattern too small for isup 5")
print(f"Good masks: {np.sum(df_train.good_mask)}/{len(df_train)} ({round(100*np.sum(df_train.good_mask)/len(df_train),1)}%)")
df_train.to_csv("/kaggle/working/train_mask_cleanup.csv")