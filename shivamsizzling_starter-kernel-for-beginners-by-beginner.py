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
train_labels = pd.read_csv("../input/train.csv")

label_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

train_labels_modified = train_labels.copy()
for i in range(train_labels_modified.shape[0]):
    train_labels_modified.Target[i] = train_labels_modified.Target[i].split()
    for k,j in enumerate(train_labels_modified.Target[i]):
        train_labels_modified.Target[i][k] = label_names[int(j)]
    
train_labels_modified.head()
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
s = pd.DataFrame(mlb.fit_transform(train_labels_modified.Target),columns=mlb.classes_, index=train_labels_modified.index)
train_labels_one_hot = train_labels_modified.join(s)
train_labels_one_hot
import seaborn as sns
import matplotlib.pyplot as plt
target_counts = train_labels_one_hot.drop(["Id", "Target"],axis =1).sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(10,10))
sns.barplot(x = target_counts.values, y=target_counts.index, order=target_counts.index)
plt.xlabel("Number of Occurences in training set")
#plt.ylabel("Protein Type")
plt.figure(figsize=(10,10))
# Using the same code as above, but divided by number of training images to get percentage
sns.barplot(x = (target_counts.values)/train_labels_modified.shape[0], y=target_counts.index, order=target_counts.index)

plt.xlabel("Fraction of Occurences in training set")
print("Number of training images with no protein identified:", train_labels_modified.Target.isnull().sum())
# every image has atleast one protein identified, so we don't have to worry about  missing data.
# Distribution of number of proteins per image

occurances  = [len(train_labels_modified.Target[i]) for i in range(train_labels_modified.shape[0])]
plt.hist(occurances, align = "left",range = [0,5])


plt.figure(figsize=(10,10))
sns.heatmap(train_labels_one_hot.drop(["Id", "Target"],axis=1).corr(),cmap="RdYlBu", vmin=-1, vmax=1)
#print(os.listdir("../input/train"))
from skimage.io import imread
def load_image(image_id, path="../input/train/"):
    images = np.zeros((4,512,512))
    images[0,:,:] = imread(path + image_id + "_green" + ".png")
    images[1,:,:] = imread(path + image_id + "_red" + ".png")
    images[2,:,:] = imread(path + image_id + "_blue" + ".png")
    images[3,:,:] = imread(path + image_id + "_yellow" + ".png")
    return images
#load_image(image_id = "000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0")
def display_image_row(image, axis, title):
    axis[0].imshow(image[0,:,:], cmap = "Greens")
    axis[1].imshow(image[1,:,:], cmap = "Reds")
    axis[2].imshow(image[2,:,:], cmap = "Blues")    
    axis[3].imshow(image[3,:,:], cmap = "Oranges")
    axis[1].set_title("microtubules")
    axis[2].set_title("nucleus")
    axis[3].set_title("endoplasmatic reticulum")