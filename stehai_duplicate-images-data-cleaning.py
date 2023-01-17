
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import collections
import imagehash
from os import path as os_path

TRAIN_IMG_PATH = r"../input/train"
###############################################################################
# misc functions

def plot_images(path, imgs):
    assert(isinstance(imgs, collections.Iterable))
    imgs_list = list(imgs)
    nrows = len(imgs_list)
    if (nrows % 2 != 0):
        nrows = nrows + 1 

    plt.figure(figsize=(18, 6*nrows/2))
    for i, img_file in enumerate(imgs_list):
        with Image.open(os_path.join(path, img_file)) as img:
            ax = plt.subplot(nrows/2, 2, i+1)
            ax.set_title("#{}: '{}'".format(i+1, img_file))
            ax.imshow(img)
        
    plt.show()


###############################################################################
# load data

def getImageMetaData(file_path):
    with Image.open(file_path) as img:
        img_hash = imagehash.phash(img)
        return img.size, img.mode, img_hash

def get_train_input():
    train_input = pd.read_csv(r"../input/train.csv")
    
    m = train_input.Image.apply(lambda x: getImageMetaData(os_path.join(TRAIN_IMG_PATH, x)))
    train_input["Hash"] = [str(i[2]) for i in m]
    train_input["Shape"] = [i[0] for i in m]
    train_input["Mode"] = [str(i[1]) for i in m]
    train_input["Length"] = train_input["Shape"].apply(lambda x: x[0]*x[1])
    train_input["Ratio"] = train_input["Shape"].apply(lambda x: x[0]/x[1])
    train_input["New_Whale"] = train_input.Id == "new_whale"
    
    return train_input

train_input = get_train_input()

###############################################################################
# data cleaning duplicate images

# determine duplicate images using the hash

t = train_input.Hash.value_counts()
t = t[t > 1]
duplicates_df = pd.DataFrame(t)

# get the Ids of the duplicate images
duplicates_df["Ids"] =list(map(
            lambda x: set(train_input.Id[train_input.Hash==x].values), 
            t.index))
duplicates_df["Ids_count"] = duplicates_df.Ids.apply(lambda x: len(x))
duplicates_df["Ids_contain_new_whale"] = duplicates_df.Ids.apply(lambda x: "new_whale" in x)

print(duplicates_df.head(20))

###
# There are 3 types of data errors regarding duplicate images:
#
# 1) The same image with the corresponding Id appears multiple time.
# 2) The same image appears with an Id and as "new_whale".
# 3) The same image appears with different Ids (ambiguous classified). 
#

# Fix error type 1: The same image with the corresponding Id appears multiple time.

train_input.drop_duplicates(["Hash", "Id"], inplace = True)

# Fix error type 2: The same image appears with an Id and as "new_whale".
# => delete the "new_whale" entry

drop_hash = duplicates_df.loc[(duplicates_df.Ids_count>1) & (duplicates_df.Ids_contain_new_whale==True)].index
train_input.drop(train_input.index[(train_input.Hash.isin(drop_hash) & (train_input.Id=="new_whale"))], inplace=True)

# Fix error type 3: The same image appears with different Ids (ambiguous classified).
# => delete all of them

drop_hash = duplicates_df.loc[(duplicates_df.Ids_count>1) & ((duplicates_df.Ids_count - duplicates_df.Ids_contain_new_whale)>1)].index

#print("Ambiguous classified images:")
#for i in drop_hash:
#    plot_images(TRAIN_IMG_PATH, 
#                train_input[train_input.Hash==i].Image)

train_input.drop(train_input.index[train_input.Hash.isin(drop_hash)], inplace=True)

# check if there are still duplicate images
assert(np.sum(train_input.Hash.value_counts()>1) == 0)
