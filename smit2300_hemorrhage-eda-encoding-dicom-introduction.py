!printf "==Data Root==\n"

!ls ../input

!printf "\n==Data Structure==\n"

!ls ../input/rsna-intracranial-hemorrhage-detection



# Alternatively if you've downloaded the dataset locally you could run something like:

# $tree -L 2 ../input
!printf "\n==Profiling Training Dataset==\n"

!ls -U ../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ | wc
!printf "\n==Training Image Filename Examples==\n"

!ls -U ../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ | head -5
!printf "\n==Labels CSV Syntax==\n"

!head -5 ../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv
!printf "\n==DCM File==\n"

!file ../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_b48228b12.dcm
import os

import random



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import pydicom

import PIL



from tqdm import tqdm, tqdm_notebook
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



print("Train dataset consists of {} images".format(num_train))

print("Test  dataset consists of {} images".format(num_test))
df_tmp = pd.read_csv(

    train_labels_path,

    nrows=5

)



columns = list(df_tmp.columns)



print("\nFeatures in training labels:")

for column in columns:

    print(column)



print("\nDataFrame Datatype Information:")

print(df_tmp.info())
%%time

def create_efficient_df(data_path):

    

    # Define the datatypes we're going to use

    final_types = {

        "ID": "str",

        "Label": "float32"

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

# test_labels_df  = create_efficient_df(test_labels_path)
train_labels_df.info()
# Syntax = Which image + hemorrhage type, Probability image contains that hemorrhage type

train_labels_df.head(10)
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

new_labels_df = pd.DataFrame(data=empty_array, columns=new_cols)



# Fill in the ID of each image

new_labels_df["id"] = list(train_labels_df.iloc[::len(hem_types)]["ID"].str.split(pat="_").str[1])

    

# Fill in the categorical columns of each image

for hem_ix, hem_col in enumerate(list(new_labels_df)[1:]):

    new_labels_df[hem_col] = list(train_labels_df.iloc[hem_ix::len(hem_types), 1])

                        

new_labels_df.sample(10)
random_ix = random.randint(0, len(train_img_paths))

random_path = train_img_root + train_img_paths[random_ix]



dcm_info = pydicom.dcmread(random_path)

print("===DICOM MEDICAL INFO===")

print(dcm_info)



pixel_data = dcm_info.pixel_array

print("\n===IMAGE PIXEL INFO===")

print("Image dimensions: {}".format(pixel_data.shape))

print(np.max(pixel_data))

print(np.min(pixel_data))

print(np.median(pixel_data))



plt.figure(figsize=(10,10))

sns.distplot(pixel_data.flatten())

plt.title("Pixel Brightness Distribution for DICOM Image")
# Function to show a random image containing a specific type of hemorrhage

# We can set the threshold to be lower or higher as well. 

def show_random_sample(hem_choice, thresh):

    

    types = new_labels_df.columns[1:]

    chosen_type = types[hem_choice]

    

    print("Displaying image with >= %.2f%% chance of containing an _%s_ hemmorhage..." % (thresh*100, chosen_type))



    filtered_df = new_labels_df[new_labels_df[chosen_type] > thresh]

    

    random_ix = random.randint(0, filtered_df.shape[0])

    

    target_record = filtered_df.iloc[random_ix, :]

    target_id = target_record[0]

    image_path = train_img_root + "ID_" + target_id + ".dcm"

    

    print("Opening {}...".format(image_path))

    

    print(target_record)

    

    dcm_img = pydicom.dcmread(image_path)

    plt.imshow(dcm_img.pixel_array)

    

    plt.grid("off")

    plt.axis("off")

    plt.title("Image of Patient with {} Hemorrhage".format(hem_types[hem_choice].title()))

    

    plt.show()
for type in range(6):

    show_random_sample(type, 0.8)
def display_by_id(patient_id):



    image_path = train_img_root + "ID_" + patient_id + ".dcm"

    

    print("Opening {}...".format(image_path))

        

    dcm_img = pydicom.dcmread(image_path)

    plt.imshow(dcm_img.pixel_array)

    

    plt.grid("off")

    plt.axis("off")

    plt.title("Image of Patient {}".format(patient_id))
display_by_id("4e16848f1")