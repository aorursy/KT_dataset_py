import os

import random



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import pydicom



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
hem_df.head()
random_ix = random.randint(0, len(train_img_paths))

random_path = train_img_root + train_img_paths[random_ix]



dcm_info = pydicom.dcmread(random_path)

print("===IMAGE MEDICAL INFO===")

print(dcm_info)
DEV_RUN = True



if DEV_RUN:

    SET_SIZE = 50_000

    print("Creating {} element subset of hemorrhage dataset".format(SET_SIZE))

    hem_df = hem_df.iloc[:SET_SIZE, :]

    

patient_ids  = np.zeros((hem_df.shape[0],))

positions    = np.zeros((hem_df.shape[0]))

orientations = np.zeros((hem_df.shape[0]))



hem_df["patient_id"]    = patient_ids

hem_df["position_0"]    = positions

hem_df["position_1"]    = positions

hem_df["position_2"]    = positions

hem_df["orientation_0"] = orientations

hem_df["orientation_1"] = orientations

hem_df["orientation_2"] = orientations

hem_df["orientation_3"] = orientations

hem_df["orientation_4"] = orientations

hem_df["orientation_5"] = orientations



del patient_ids

del positions

del orientations



for row_ix, row in tqdm_notebook(hem_df.iterrows()):

    

    full_path = train_img_root + "ID_" + row["id"] + ".dcm"

    dcm_info  = pydicom.dcmread(full_path)

    

    patient_id  = dcm_info.PatientID.split("_")[1]

    position    = dcm_info.ImagePositionPatient

    orientation = dcm_info.ImageOrientationPatient

        

    hem_df["patient_id"].iloc[row_ix]  = patient_id

    

    hem_df["position_0"].iloc[row_ix]    = position[0]

    hem_df["position_1"].iloc[row_ix]    = position[1]

    hem_df["position_2"].iloc[row_ix]    = position[2]

    

    hem_df["orientation_0"].iloc[row_ix] = orientation[0]

    hem_df["orientation_1"].iloc[row_ix] = orientation[1]

    hem_df["orientation_2"].iloc[row_ix] = orientation[2]

    hem_df["orientation_3"].iloc[row_ix] = orientation[3]

    hem_df["orientation_4"].iloc[row_ix] = orientation[4]

    hem_df["orientation_5"].iloc[row_ix] = orientation[5]

        

hem_df.head()
dup_df = hem_df.pivot_table(index=['patient_id'], aggfunc='size')

dup_df = dup_df[dup_df > 1]



patient_df = hem_df[hem_df["patient_id"] == dup_df.idxmax()]

patient_df = patient_df.sort_values("id")



print("=======PATIENT ID: {}=======".format(patient_df["patient_id"].iloc[0]))



def show_patient_frames(df):

    

    id_list = list(df["id"])



    # Used for subplots but that's been deprecated for larger subset sizes

    num_cols = 3

    num_rows = int(len(id_list) / num_cols)

    

    id_ix = 0

    for row in range(num_rows):

        for col in range(num_cols):

            

            fig = plt.figure(figsize=(8,8))

    

            current_id = id_list[id_ix]

            full_path = train_img_root + "ID_" + current_id + ".dcm"

            dcm_info = pydicom.dcmread(full_path)

            pixel_data = dcm_info.pixel_array



            plt.imshow(pixel_data)



            plt.grid("off")

            plt.axis("off")

#             axes[row, col].set_title("Image ID: {}\nEpidural: {}\nIntraparenchymal: {}\nIntraventricular: {}\nSubdural: {}\nSubarachnoid: {}"

#                  .format(current_id, df.iloc[id_ix, 1], df.iloc[id_ix, 2], df.iloc[id_ix, 3], df.iloc[id_ix, 4], df.iloc[id_ix, 5]))



            plt.title("Image ID: {}\nx: {} y: {} z: {}"

                    .format(current_id, df.iloc[id_ix, 8], df.iloc[id_ix, 9], df.iloc[id_ix, 10]))



            id_ix += 1



    plt.show()

    

show_patient_frames(patient_df)