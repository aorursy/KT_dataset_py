import os

import random



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import pydicom



from skimage.measure import regionprops, label

from skimage.segmentation import mark_boundaries



from tqdm import tqdm_notebook
# Get directory names/locations

data_root = os.path.abspath("../input/rsna-intracranial-hemorrhage-detection/")



train_img_root = data_root + "/stage_1_train_images/"

test_img_root  = data_root + "/stage_1_test_images/"



train_labels_path = data_root + "/stage_1_train.csv"

test_labels_path  = data_root + "/stage_1_test.csv"



# Create list of paths to actual training data

train_img_paths = os.listdir(train_img_root)

test_img_paths  = os.listdir(test_img_root)



# Dataset size

num_train = len(train_img_paths)

num_test  = len(test_img_paths)
def create_efficient_df(data_path):

    

    # Define the datatypes we're going to use

    final_types = {

        "ID": "str",

        "Label": "float16"

    }

    features = list(final_types.keys())

    

    # Use chunks to import the data so that less efficient machines can only use a 

    # specific amount of chunks on import

    df_list = []



    chunksize = 1_000_000



    for df_chunk in pd.read_csv(data_path, dtype=final_types, chunksize=chunksize): 

        df_list.append(df_chunk)

        

    df = pd.concat(df_list)

    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]



    del df_list



    return df



train_labels_df = create_efficient_df(train_labels_path)

train_labels_df[train_labels_df["Label"] > 0].head()
hem_types = [

    "epidural",

    "intraparenchymal",

    "intraventricular",

    "subarachnoid",

    "subdural",

    "any"

]



new_cols = [

    "id",

    "type_0",

    "type_1",

    "type_2",

    "type_3",

    "type_4",

    "type_5"

]



num_ids = int(train_labels_df.shape[0] / len(hem_types))

print("Number of unique patient IDs: {}".format(num_ids))



empty_array = np.ones((num_ids, len(new_cols)))

hem_df = pd.DataFrame(data=empty_array, columns=new_cols)



# Fill in the ID of each image

hem_df["id"] = list(train_labels_df.iloc[::len(hem_types)]["ID"].str.split(pat="_").str[1])

    

# Fill in the categorical columns of each image

for hem_ix, hem_col in enumerate(list(hem_df)[1:]):

    hem_df[hem_col] = list(train_labels_df.iloc[hem_ix::len(hem_types), 1])

    

hem_df.info()

hem_df[hem_df["type_5"] > 0].head()
CERTAINTY = 0.95



# Filter the dataframe to search for epidural hemorrhages

epi_df = hem_df[(hem_df["type_0"] > CERTAINTY) & (hem_df["type_1"] < CERTAINTY) & (hem_df["type_2"] < CERTAINTY) & (hem_df["type_3"] < CERTAINTY) & (hem_df["type_4"] < CERTAINTY)]



# Custom indices of images that contain good looking hemorrhages to me (please suggest better image if anyone know of any!)

epi_ix = 6



# Slice out the record at the chosen index

epi_record = epi_df.iloc[epi_ix, :]



# Get the image path from the record

epi_path = train_img_root + "ID_" + epi_record["id"] + ".dcm"



# Use PyDICOM to open the image and get array data

epidural_frame = pydicom.dcmread(epi_path).pixel_array



# Normalize the array between 0 and 255

epidural_frame = np.interp(epidural_frame, (epidural_frame.min(), epidural_frame.max()), (0, 255))
def plot_frame(img_array, cmap, title):

    plt.figure(figsize=(8,8))

    plt.imshow(img_array, cmap=cmap)

    plt.title(title, fontsize=16)

    plt.axis("off")

    

plot_frame(epidural_frame, "bone", "CT Scan of Epidural Hemorrhage")

plot_frame(epidural_frame, "hot", "CT Scan of Epidural Hemorrhage")

plot_frame(epidural_frame, "viridis", "CT Scan of Epidural Hemorrhage")
sns.distplot(epidural_frame.flatten())
dense_mask   = (epidural_frame > 200).astype(int)

dense_frame  = (epidural_frame * dense_mask).astype(int)

dense_coords = np.argwhere(dense_frame)



plot_frame(epidural_frame, "viridis", "Original Epidural Hemorrhage Image")

plot_frame(dense_mask, "hot", "Region of High Density Segmentation Mask")

plot_frame(dense_frame, "viridis", "Regions of High Density")
norm_frame = epidural_frame / 255.0

segmented_frame = mark_boundaries(norm_frame, dense_mask, color=(255,0,0))



plot_frame(segmented_frame, "bone", "Marked Boundaries of High Density")
label_img = label(dense_mask)

regions   = regionprops(label_image=label_img, intensity_image=dense_frame)



largest_area = max([regions[x].area for x in range(len(regions))])

print("Largest high density area: {}".format(largest_area))



num_bones = len(regions)

print("Number of high density regions: {}".format(num_bones))



for prop_ix, props in enumerate(regions):

    print("\nHotspot number {}:".format(prop_ix+1))

    print("Area: %d" % (props.area))

    print("Centroid: (%.2f, %.2f)" % (props.centroid[0],props.centroid[1]))

    print("Mean Intensity: %.2f" % (props.mean_intensity))

    print("Max Intensity: %d" % (props.max_intensity))

    