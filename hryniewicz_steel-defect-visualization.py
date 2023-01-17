from collections import Counter

from pathlib import Path



import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
input_dir = Path("../input/severstal-steel-defect-detection/")

df = pd.read_csv(input_dir / "train.csv")
df["ImageId"] = df.ImageId_ClassId.str[:-2]

df["ClassId"] = df.ImageId_ClassId.str[-1]

df = df[["ImageId", "ClassId", "EncodedPixels"]]

images = df.ImageId.unique()

classes = df.ClassId.unique()

df.head()
class_count = [df[df.ClassId == c].EncodedPixels.notnull().sum() for c in classes]

plt.bar(classes, class_count)

plt.show()
defects_imgs = {"0_defects": [],

                "2_defects": [],

                "3_defects": []}

for k in classes:

    defects_imgs[f"class_{k}"] = []

for i in images:

    defects = np.where(df[df.ImageId==i].EncodedPixels.notnull())[0]

    defects_count = len(defects)

    if defects_count == 1:

        defects_imgs[f"class_{defects[0]+1}"].append(i)

    else:

        defects_imgs[f"{defects_count}_defects"].append(i)



print(list(zip(defects_imgs.keys(), map(len, defects_imgs.values()))))
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

image_dir = input_dir / "train_images"

img_shape = cv2.imread(str(image_dir / df.iloc[0].ImageId)).shape[:2]

pixel_count = np.prod(img_shape)
def show_with_masks(image_name):

    # build masks

    masks = np.zeros((len(classes),) + img_shape, dtype=np.uint8)

    for __, row in df[df.ImageId == image_name].iterrows():

        if not pd.isna(row.EncodedPixels):

            # decode pixels to a mask

            encoded = row.EncodedPixels.split()

            position = map(int, encoded[0::2])

            run_length = map(int, encoded[1::2])

            mask = np.zeros(pixel_count, dtype=np.uint8)

            for p, l in zip(position, run_length):

                mask[p-1:p+l-1] = 1

            masks[int(row.ClassId)-1] = mask.reshape(img_shape, order='F')

    # put counters on the image

    img = cv2.imread(str(image_dir / image_name))

    for i, mask in enumerate(masks):

        contours, __ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for contour in contours:

            cv2.polylines(img, contour, True, colors[i], 2)

    # show result

    __, ax = plt.subplots(figsize=(30,5))

    ax.imshow(img)

    plt.axis("off")

    plt.show()
for k, v in defects_imgs.items():

    print(f"{k} ({len(v)} images)")

    for i in range(0, min(3, len(v))):

        show_with_masks(v[i])