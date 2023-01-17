# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import re
from itertools import product

# matplotlib style
plt.style.use('fivethirtyeight')

# random state
RSTATE=1984
# color hunt palettes
ch_div_palette_1 = ["#288fb4", "#1d556f", "#efddb2", "#fa360a"]
ch_div_palette_2 = ["#ff5335", "#dfe0d4", "#3e92a3", "#353940"]
ch_div_palette_3 = ["#daebee", "#b6d7de", "#fcedda", "#ff5126"]
# matplotlib "fivethirtyeight" style colors
ch_div_palette_4 = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c']
# https://www.kaggle.com/c/human-protein-atlas-image-classification/data
label_names = [
    "Nucleoplasm",
    "Nuclear membrane",
    "Nucleoli",
    "Nucleoli fibrillar center",
    "Nuclear speckles",
    "Nuclear bodies",
    "Endoplasmic reticulum",
    "Golgi apparatus",
    "Peroxisomes",
    "Endosomes",
    "Lysosomes",
    "Intermediate filaments",
    "Actin filaments",
    "Focal adhesion sites",
    "Microtubules",
    "Microtubule ends",
    "Cytokinetic bridge",
    "Mitotic spindle",
    "Microtubule organizing center",
    "Centrosome",
    "Lipid droplets",
    "Plasma membrane",
    "Cell junctions",
    "Mitochondria",
    "Aggresome",
    "Cytosol",
    "Cytoplasmic bodies",
    "Rods & rings",    
]
    
def get_num_labels_for_instance(label_string):
    labels = re.split(r'\s+', label_string)
    return len(labels)

def get_label_presence_func(label):
    def is_label_present(label_string):
        labels = set(re.split(r'\s+', label_string))
        return int(str(label) in labels)
    return is_label_present

def is_single_label(label_string):
    label_ids = re.split(r'\s+', label_string)
    if len(label_ids) > 1:
        return False
    return True
 
def get_label_name_for_label_id_string(label_ids_str):
    label_ids = re.split(r'\s+', label_ids_str)
    label_ids = [int(id) for id in label_ids]
    label = "+".join([label_names[id] for id in label_ids])
    return label

# Returns different bar color for single and multi-labels
def get_bar_color_1(is_single_label):
    if is_single_label:
        return ch_div_palette_1[0]
    return ch_div_palette_1[2]

# Returns different bar color for single and multi-labels
def get_bar_color_2(is_single_label):
    if is_single_label:
        return ch_div_palette_2[0]
    return ch_div_palette_2[2]

# Returns different bar color for single and multi-labels
def get_bar_color_3(is_single_label):
    if is_single_label:
        return ch_div_palette_3[2]
    return ch_div_palette_3[3]

# Returns different bar color for single and multi-labels
def get_bar_color_4(is_single_label):
    if is_single_label:
        return ch_div_palette_4[0]
    return ch_div_palette_4[1]
labels_df = pd.read_csv("../input/train.csv")
print("Shape of the training labels frame (train.csv): ", labels_df.shape)
labels_df.head()
labels_df["num_labels"] = labels_df["Target"].apply(get_num_labels_for_instance)
labels_count_dist = labels_df.groupby("num_labels")["num_labels"].count()
fig, ax = plt.subplots(num=1)
ax.bar(labels_count_dist.index.values, labels_count_dist.values)
ax.set_xlabel("Number of labels per instance")
ax.set_ylabel("Number of instances")
ax.set_title("Labels count distribution")
plt.show()
multi_labels_dist = pd.DataFrame()
tmp = labels_df.groupby("Target")["Target"].count().sort_values(ascending=False)
multi_labels_dist["Target"] = tmp.index.values
multi_labels_dist["Count"] = tmp.values
multi_labels_dist["is_single_label"] = multi_labels_dist["Target"].apply(is_single_label)
multi_labels_dist["Target_str"] = multi_labels_dist["Target"].apply(get_label_name_for_label_id_string)
multi_labels_dist = multi_labels_dist[["Target", "Target_str", "Count", "is_single_label"]]
print("Number of unique labels (single/multi): {}".format(multi_labels_dist.shape[0]))
multi_labels_dist.head()
topn = 50
fig, ax = plt.subplots(num=2)
fig.set_figwidth(15)
fig.set_figheight(10)
bar_colors = multi_labels_dist["is_single_label"].apply(get_bar_color_4).head(topn)
ax.bar(multi_labels_dist["Target_str"].head(topn), multi_labels_dist["Count"].head(topn), color=bar_colors)
ax.set_xticks(range(topn))
ax.set_xticklabels(multi_labels_dist["Target_str"].head(topn), rotation = 45, ha="right")
ax.set_title("Distribution of top-{} training labels (single/multi)".format(topn))
plt.show()
label_columns_df = pd.DataFrame()
for i in range(len(label_names)):
    label_chk_fn = get_label_presence_func(i)
    label_columns_df[label_names[i]] = labels_df["Target"].apply(label_chk_fn)
labels_dist = label_columns_df.sum().sort_values(ascending=False)
fig, ax = plt.subplots(num=2)
fig.set_figwidth(15)
fig.set_figheight(10)
ax.bar(labels_dist.index.values, labels_dist.values)
ax.set_xticks(range(len(labels_dist.values)))
ax.set_xticklabels(labels_dist.index.values, rotation = 45, ha="right")
ax.set_title("Distribution of single labels")
plt.show()
fig, ax = plt.subplots(num=1, nrows=3, ncols=3)
fig.set_figheight(15)
fig.set_figwidth(15)
for idx, (x, y) in enumerate(product(range(3), range(3))):
    img_blue = cv2.imread("../input/train/" + labels_df.loc[idx, "Id"] + "_blue.png", cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread("../input/train/" + labels_df.loc[idx, "Id"] + "_green.png", cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread("../input/train/" + labels_df.loc[idx, "Id"] + "_red.png", cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.merge((img_blue, img_green, img_red))
    image_label = get_label_name_for_label_id_string(labels_df.loc[idx, "Target"])
    ax[x,y].imshow(img_bgr)
    ax[x,y].set_xticks([])
    ax[x,y].set_yticks([])
    ax[x,y].set_title(image_label, fontdict={"fontsize": 12})
plt.show()
# top-5
topn = 5
topn_labels = multi_labels_dist.head(topn)["Target"]
multi_labels_dist.head(5)
images_per_label = 5
fig, ax = plt.subplots(num=1, nrows=topn, ncols=images_per_label)
fig.set_figheight(21)
fig.set_figwidth(21)
for idx, t in enumerate(topn_labels):
    sample_ids_for_label = labels_df[labels_df["Target"] == t].sample(n=images_per_label, random_state=RSTATE)["Id"].tolist()
    ax[idx, 0].set_ylabel(get_label_name_for_label_id_string(t))
    for idy in range(images_per_label):
        img_blue = cv2.imread("../input/train/" + sample_ids_for_label[idy] + "_blue.png", cv2.IMREAD_GRAYSCALE)
        img_green = cv2.imread("../input/train/" + sample_ids_for_label[idy] + "_green.png", cv2.IMREAD_GRAYSCALE)
        img_red = cv2.imread("../input/train/" + sample_ids_for_label[idy] + "_red.png", cv2.IMREAD_GRAYSCALE)
        img_bgr = cv2.merge((img_blue, img_green, img_red))
        ax[idx,idy].imshow(img_bgr)
        ax[idx,idy].set_xticks([])
        ax[idx,idy].set_yticks([])
        ax[idx,idy].set_title(sample_ids_for_label[idy], fontdict={"fontsize":10})
plt.show()
n_images = 25
ncols = 5
nrows = n_images // ncols
image_ids = labels_df[labels_df["Target"] == "0"].sample(n=n_images, random_state=RSTATE)["Id"].tolist()
fig, ax = plt.subplots(num=1, nrows=nrows, ncols=ncols)
fig.set_figheight(21)
fig.set_figwidth(21)
for idx, (x, y) in enumerate(product(range(nrows), range(ncols))):
    img_blue = cv2.imread("../input/train/" + image_ids[idx] + "_blue.png", cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread("../input/train/" + image_ids[idx] + "_green.png", cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread("../input/train/" + image_ids[idx] + "_red.png", cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.merge((img_blue, img_green, img_red))
    ax[x,y].imshow(img_bgr)
    ax[x,y].set_xticks([])
    ax[x,y].set_yticks([])
    ax[x,y].set_title(image_ids[idx], fontdict={"fontsize": 12})
plt.show()
n_images = 25
ncols = 5
nrows = n_images // ncols
image_ids = labels_df[labels_df["Target"] == "23"].sample(n=n_images, random_state=RSTATE)["Id"].tolist()
fig, ax = plt.subplots(num=1, nrows=nrows, ncols=ncols)
fig.set_figheight(21)
fig.set_figwidth(21)
for idx, (x, y) in enumerate(product(range(nrows), range(ncols))):
    img_blue = cv2.imread("../input/train/" + image_ids[idx] + "_blue.png", cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread("../input/train/" + image_ids[idx] + "_green.png", cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread("../input/train/" + image_ids[idx] + "_red.png", cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.merge((img_blue, img_green, img_red))
    ax[x,y].imshow(img_bgr)
    ax[x,y].set_xticks([])
    ax[x,y].set_yticks([])
    ax[x,y].set_title(image_ids[idx], fontdict={"fontsize": 12})
plt.show()
n_images = 25
ncols = 5
nrows = n_images // ncols
image_ids = labels_df[labels_df["Target"] == "25"].sample(n=n_images, random_state=RSTATE)["Id"].tolist()
fig, ax = plt.subplots(num=1, nrows=nrows, ncols=ncols)
fig.set_figheight(21)
fig.set_figwidth(21)
for idx, (x, y) in enumerate(product(range(nrows), range(ncols))):
    img_blue = cv2.imread("../input/train/" + image_ids[idx] + "_blue.png", cv2.IMREAD_GRAYSCALE)
    img_green = cv2.imread("../input/train/" + image_ids[idx] + "_green.png", cv2.IMREAD_GRAYSCALE)
    img_red = cv2.imread("../input/train/" + image_ids[idx] + "_red.png", cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.merge((img_blue, img_green, img_red))
    ax[x,y].imshow(img_bgr)
    ax[x,y].set_xticks([])
    ax[x,y].set_yticks([])
    ax[x,y].set_title(image_ids[idx], fontdict={"fontsize": 12})
plt.show()
