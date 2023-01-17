import pandas as pd

import numpy as np

import os

from PIL import Image, ImageDraw

from ast import literal_eval

import matplotlib.pyplot as plt
!ls ../input/global-wheat-detection
root_path = "../input/global-wheat-detection/"

train_folder = os.path.join(root_path, "train")

test_folder = os.path.join(root_path, "test")

train_csv_path = os.path.join(root_path, "train.csv")

sample_submission = os.path.join(root_path, "sample_submission.csv")
df = pd.read_csv(train_csv_path)
df.head()
df.shape[0]
df['width'].unique() == df['height'].unique() == [1024]
def get_bbox_area(bbox):

    bbox = literal_eval(bbox)

    return bbox[2] * bbox[3]
df['bbox_area'] = df['bbox'].apply(get_bbox_area)
df['bbox_area'].value_counts().hist(bins=50)
unique_images = df['image_id'].unique()
num_total = len(os.listdir(train_folder))

num_annotated = len(unique_images)



print(f"There are {num_annotated} annotated images and {num_total - num_annotated} images without annotations.")
sources = df['source'].unique()

print(f"There are {len(sources)} sources of data: {sources}")
df['source'].value_counts()
plt.hist(df['image_id'].value_counts(), bins=30)

plt.show()
def show_images(images, num = 5):

    

    images_to_show = np.random.choice(images, num)



    for image_id in images_to_show:



        image_path = os.path.join(train_folder, image_id + ".jpg")

        image = Image.open(image_path)



        # get all bboxes for given image in [xmin, ymin, width, height]

        bboxes = [literal_eval(box) for box in df[df['image_id'] == image_id]['bbox']]



        # visualize them

        draw = ImageDraw.Draw(image)

        for bbox in bboxes:    

            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width=3)



        plt.figure(figsize = (15,15))

        plt.imshow(image)

        plt.show()
show_images(unique_images)
for source in sources:

    print(f"Showing images for {source}:")

    show_images(df[df['source'] == source]['image_id'].unique(), num = 3)