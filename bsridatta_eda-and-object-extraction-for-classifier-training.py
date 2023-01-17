import numpy as np

import pandas as pd 

import os

import json

import glob

import random

from IPython.display import display, display_markdown

from math import floor

from matplotlib import pyplot as plt

import matplotlib.image as mpimg

import matplotlib.patches as patches

from mpl_toolkits.axes_grid1 import ImageGrid

import seaborn as sns

from PIL import Image

from tqdm.notebook import tqdm
img = "/kaggle/input/iwildcam-2020-fgvc7/train/92b8a1d0-21bc-11ea-a13a-137349068a90.jpg"

_ = plt.figure(figsize = (15,20))

_ = plt.axis('off')

_ = plt.imshow(mpimg.imread(img)[100:-100])
os.listdir("/kaggle/input/iwildcam-2020-fgvc7")
print("Number of train images: ", len(glob.glob(f'/kaggle/input/iwildcam-2020-fgvc7/train/*')))

print("Number of test images: ", len(glob.glob(f'/kaggle/input/iwildcam-2020-fgvc7/test/*')))
with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as json_data:

    train_annotations = json.load(json_data)

    print(train_annotations.keys())
df_cat = pd.DataFrame(train_annotations["categories"])

display(f"Total Categories: {df_cat.name.nunique()}")

display(df_cat.sample(5))
display("Samples of annotations and images")

df_train_annotations = pd.DataFrame(train_annotations["annotations"])

display(df_train_annotations.sample())
with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_megadetector_results.json') as json_data:

    megadetector_results = json.load(json_data)

    print(megadetector_results.keys())

print(megadetector_results['info'])
df_detections = pd.DataFrame(megadetector_results["images"])

print(f'detection categories :\n {megadetector_results["detection_categories"]}')

print(f'detection output :\n {df_detections.head()}')
with open('/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_test_information.json') as json_data:

    test_info = json.load(json_data)

    print(test_info.keys())
print(f'test images :\n {test_info["images"][0]}')
train = glob.glob(f'/kaggle/input/iwildcam-2020-fgvc7/train/*')

print("Train Path \n", train[0])

test = glob.glob(f'/kaggle/input/iwildcam-2020-fgvc7/test/*')

print("Test Path \n", test[0])
def plot_images(rows,cols):

    fig = plt.figure(figsize=(15., 12.))

    grid = ImageGrid(fig, 111,  # similar to subplot(111)

                 nrows_ncols=(rows, cols),  # creates 5x5 grid of axes

                 axes_pad=0.3,  # pad between axes in inch.

                 )



    for ax, img in zip(grid, random.sample(train, rows*cols)):

        image_id = img.split('/')[-1].split('.')[0]

        cat_id = df_train_annotations[df_train_annotations.image_id == image_id].category_id

        cat = df_cat[df_cat.id == int(cat_id)].name.values[0]

        # Iterating over the grid returns the Axes.

        _ = ax.set_title(str(cat))

        _ = ax.imshow(mpimg.imread(img))

        _ = ax.axis('off')



    _ = plt.show()
plot_images(3,3)
def draw_bbox(img_path = "/kaggle/input/iwildcam-2020-fgvc7/train/92b8a1d0-21bc-11ea-a13a-137349068a90.jpg"):

    

    img_id = img_path.split('/')[-1].split('.')[0] 

    img = mpimg.imread(img_path)

    detections = df_detections[df_detections.id==img_id].detections.values[0]

    annotation = df_train_annotations[df_train_annotations.image_id == img_id]

    

    count = annotation['count'].values

    cat_id = annotation.category_id

    cat = df_cat[df_cat.id == int(cat_id)].name.values[0]

    

    _ = plt.figure(figsize = (15,20))

    _ = plt.axis('off')

    ax = plt.gca()

    ax.text(10,100, f'{cat} {count}', fontsize=20, color='fuchsia')



    for detection in detections:

        # ref - https://github.com/microsoft/CameraTraps/blob/e530afd2e139580b096b5d63f0d7ab9c91cbc7a4/visualization/visualization_utils.py#L392

        x_rel, y_rel, w_rel, h_rel = detection['bbox']    

        img_height, img_width, _ = img.shape

        x = x_rel * img_width

        y = y_rel * img_height

        w = w_rel * img_width

        h = h_rel * img_height

        

        cat = 'animal' if detection['category'] == "1" else 'human'

        bbox = patches.FancyBboxPatch((x,y), w, h, alpha=0.8, linewidth=6, capstyle='projecting', edgecolor='fuchsia', facecolor="none")

        

        ax.text(x+1.5, y-8, f'{cat} {detection["conf"]}', fontsize=10, bbox=dict(facecolor='fuchsia', alpha=0.8, edgecolor="none"))

        ax.add_patch(bbox)



    _ = plt.imshow(img)
img_path = "/kaggle/input/iwildcam-2020-fgvc7/train/92b8a1d0-21bc-11ea-a13a-137349068a90.jpg"

draw_bbox(img_path)
df_train_annotations.category_id.value_counts()
plt.figure(figsize=(40,5))

df_cat_dist = df_train_annotations.category_id.value_counts()

print(f"Excluding {df_cat_dist[0]} images from the empty class in the barplot visualization")

df_cat_dist = df_cat_dist[1:]

chart = sns.barplot(y=df_cat_dist.values, x=df_cat_dist.index, orient='v')

_ = chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
def extract_objects(img_path = "/kaggle/input/iwildcam-2020-fgvc7/train/92b8a1d0-21bc-11ea-a13a-137349068a90.jpg", show=False):

    objects = []

    confidences = []

    categories = []

    

    img_id = img_path.split('/')[-1].split('.')[0] 

    img = np.array(mpimg.imread(img_path))

    if (df_detections[df_detections.id==img_id].detections.values):

        pass

    else:

        return None

    detections = df_detections[df_detections.id==img_id].detections.values[0]

    annotation = df_train_annotations[df_train_annotations.image_id == img_id]

    cat_id = annotation.category_id

    cat = df_cat[df_cat.id == int(cat_id)].name.values[0]

    

    for idx, detection in enumerate(detections):

        # save confidence

        confidences.append(detection["conf"])

        if detection['category'] == "1":

            categories.append(cat)

        else:

            categories.append('human')



        x_rel, y_rel, w_rel, h_rel = detection['bbox']    

        img_height, img_width, _ = img.shape

        x = floor(x_rel * img_width)

        y = floor(y_rel * img_height)

        w = floor(w_rel * img_width)

        h = floor(h_rel * img_height)



        obj = img[int(y):int(y+h),int(x):int(x+w)]

        objects.append(obj)

        if show:

            _ = plt.figure()

            _ = plt.xticks([])

            _ = plt.yticks([])

            _ = plt.imshow(obj)

    

    return objects, categories, confidences
_ = extract_objects(show=True)  
def save_data(img_path):

    img_id = img_path.split('/')[-1].split('.')[0] 

    objects, cats, confs = extract_objects(img_path)

    for i in range(len(objects)):

        meta_df.loc[len(meta_df)] = [f'{img_id}_{i}', img_id, cats[i], confs[i]]

        try:

            mpimg.imsave(f'train/{img_id}_{i}.jpg', objects[i])

        except:

            pass