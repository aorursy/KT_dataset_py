import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from scipy.misc import imread

import tensorflow as tf
sns.set()

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Any results you write to the current directory are saved as output.
train_labels = pd.read_csv("../input/human-protein-atlas-image-classification/train.csv")
train_labels.head()
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

reverse_train_labels = dict((v,k) for k,v in label_names.items())

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = label_names[int(num)]
        row.loc[name] = 1
    return row
for key in label_names.keys():
    train_labels[label_names[key]] = 0
train_labels = train_labels.apply(fill_targets, axis=1)
train_labels.head()
# path to directory where training images are held
train_path = '../input/human-protein-atlas-image-classification/train/'
for color in ['red', 'green', 'blue', 'yellow']:
    train_labels[f'{color}_filename'] = \
        (train_labels['Id']
         .transform(
             lambda id: f'{id}_{color}.png'
         )
        )
    train_labels[f'{color}_path'] = \
        (train_labels[f'{color}_filename']
         .transform(
             lambda path: os.path.join(train_path, path)
         )
        )
#     Can't load all into memory at once, unfortunately
#     train_labels[f'{color}_img'] = (train_labels[f'{color}_path']
#                                     .transform(
#                                         lambda path: Image.open(path)
#                                     )
#                                    )
train_labels.head()
plt.figure(figsize=(15,15))
# Allunia's notebook had an additional slice for train_labels.number_of_targets>1,
# but then we miss the fact that Endosomes and Lysosomes are not fully correlated
# because some images are only tagged "Lysosome"
labels = [label for label in label_names.values()]
sns.heatmap(train_labels[labels].corr(),
            cmap="RdYlBu",
            vmin=-1,
            vmax=1)
plt.figure(figsize=(15,15))
train_labels["number_of_targets"] = train_labels.drop(["Id", "Target"],axis=1).sum(axis=1)
sns.heatmap(train_labels[train_labels.number_of_targets>1][labels].corr(),
            cmap="RdYlBu",
            vmin=-1,
            vmax=1)
def find_counts(special_target, labels):
    counts = labels[labels[special_target] == 1][[label for label in label_names.values()]].sum(axis=0)
    counts = counts[counts > 0]
    counts = counts.sort_values()
    return counts
lyso_counts = find_counts("Lysosomes", train_labels)

plt.figure(figsize=(10,3))
sns.barplot(x=lyso_counts.index.values, y=lyso_counts.values, palette="Blues")
endo_counts = find_counts("Endosomes", train_labels)

plt.figure(figsize=(10,3))
sns.barplot(x=endo_counts.index.values, y=endo_counts.values, palette="Blues")
mask = (train_labels["Endosomes"]==1) & (train_labels["Lysosomes"]==0)
example_index = train_labels[mask].index[0]
r = np.array(Image.open(train_labels.loc[example_index, "red_path"]))
g = np.array(Image.open(train_labels.loc[example_index, "green_path"]))
b = np.array(Image.open(train_labels.loc[example_index, "blue_path"]))
y = np.array(Image.open(train_labels.loc[example_index, "yellow_path"]))
fig, ax = plt.subplots(1,4)

ax[0].imshow(r, cmap=plt.cm.Reds)
ax[0].set_axis_off()

ax[1].imshow(g, cmap=plt.cm.Greens)
ax[1].set_axis_off()

ax[2].imshow(b, cmap=plt.cm.Blues)
ax[2].set_axis_off()

ax[3].imshow(y, cmap=plt.cm.Oranges)
ax[3].set_axis_off()
