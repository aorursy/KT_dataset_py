import os

import sys

import re

import json

from glob import glob



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

sns.set()



import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

from plotly.colors import DEFAULT_PLOTLY_COLORS as colors

py.init_notebook_mode(connected=True)



import ipywidgets as widgets

from ipywidgets import interact, interact_manual

from IPython.display import display, HTML



SMALL_SIZE = 14

MEDIUM_SIZE = 16

BIGGER_SIZE = 24



plt.rc('font', size=SMALL_SIZE)

plt.rc('axes', titlesize=BIGGER_SIZE)

plt.rc('axes', labelsize=MEDIUM_SIZE)

plt.rc('xtick', labelsize=SMALL_SIZE)

plt.rc('ytick', labelsize=SMALL_SIZE)

plt.rc('legend', fontsize=SMALL_SIZE)

plt.rc('figure', titlesize=BIGGER_SIZE)



pd.options.display.max_columns = None
!ls ../input
!ls ../input/train_images | head -3
!ls ../input/train_metadata | head -3
# read data

df_train = pd.read_csv("../input/train/train.csv")

df_breed = pd.read_csv("../input/breed_labels.csv")

df_color = pd.read_csv("../input/color_labels.csv")

df_state = pd.read_csv("../input/state_labels.csv")



# cleaning

df_train.loc[df_train["Name"].isnull(), "Name"] = np.nan



df_train.loc[df_train["Description"].isnull(), "Description"] = ""



df_train["PhotoAmt"] = df_train["PhotoAmt"].astype(int)



is_breed1_zero = df_train["Breed1"] == 0

df_train["Breed1"][is_breed1_zero] = df_train["Breed2"][is_breed1_zero]

df_train["Breed2"][is_breed1_zero] = 0



# merge dataframes

df_breed = df_breed.append({"BreedID": 0, "Type": 0, "BreedName": ""}, ignore_index=True).replace("", np.nan)

df_color = df_color.append({"ColorID":0, "ColorName": ""}, ignore_index=True).replace("", np.nan)



# Decode categorical features

df_train["Breed1"] = df_breed.set_index("BreedID").loc[df_train["Breed1"]]["BreedName"].values

df_train["Breed2"] = df_breed.set_index("BreedID").loc[df_train["Breed2"]]["BreedName"].values

df_train["Color1"] = df_color.set_index("ColorID").loc[df_train["Color1"]]["ColorName"].values

df_train["Color2"] = df_color.set_index("ColorID").loc[df_train["Color2"]]["ColorName"].values

df_train["Color3"] = df_color.set_index("ColorID").loc[df_train["Color3"]]["ColorName"].values

df_train["State"] = df_state.set_index("StateID").loc[df_train["State"]]["StateName"].values



mapdict = {

    "Type"        : ["", "Dog", "Cat"],

    "Gender"      : ["", "Male", "Female", "Mixed"],

    "MaturitySize": ["Not Specified", "Small", "Meidum", "Large", "Extra Large"],

    "FurLength"   : ["Not Specified", "Short", "Medium", "Long"],

    "Vaccinated"  : ["", "Yes", "No", "Not Sure"],

    "Dewormed"    : ["", "Yes", "No", "Not Sure"],

    "Sterilized"  : ["", "Yes", "No", "Not Sure"],

    "Health"      : ["Not Specified", "Healthy", "Minor Injury", "Serious Injury"]

}



for k, v in mapdict.items():

    dummy_df = pd.DataFrame({k: v})

    df_train[k] = dummy_df.loc[df_train[k]][k].values
def read_json(fpath):

    with open(fpath) as f:

        return json.load(f)



def get_sentiment(pet_id, dir_):

    fpath = f"../input/{dir_}/{pet_id}.json"

    if not os.path.exists(fpath):

        return np.nan, np.nan

    data = read_json(fpath)

    result = data["documentSentiment"]

    return result["magnitude"], result["score"]



def get_image_meta(pet_id, dir_):

    fpath = f"../input/{dir_}/{pet_id}-1.json"

#     print(fpath)

    if not os.path.exists(fpath):

        return np.nan, np.nan

    

    data = read_json(fpath)

    

    if not "labelAnnotations" in data:

        return np.nan, np.nan

    

    result = data["labelAnnotations"][0]

    return result["description"], result["score"]
# merge image metadata

df_train["ImageDescription"], df_train["ImageDescriptionScore"] = zip(*df_train["PetID"].map(lambda pet_id: get_image_meta(pet_id, "train_metadata")))

df_train.sample()
def rand_pet_id():

    return df_train["PetID"].sample(1).values[0]



def grouped(iterable, n):

    return zip(*[iter(iterable)]*n)



def show_pics(pet_id):

    img_paths = glob(f"../input/train_images/{pet_id}*.jpg")

    npics = len(img_paths)

    if npics == 0:

        print("No picture found")

        return

    max_ncols = 5

    ncols = max_ncols if npics > max_ncols else npics

    nrows = int(np.ceil(npics / max_ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, nrows * 4))        

    axes = [axes] if npics == 1 else axes.ravel()

    for ax in axes[npics:]: fig.delaxes(ax)

    

    for i, img_path, ax in zip(range(npics), img_paths, axes):

        ax.imshow(plt.imread(img_path))

        ax.set_xticks([])

        ax.set_yticks([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)



def show_pet(pet_id):

    row_match = df_train[df_train["PetID"] == pet_id]

    d = {col: ser.iloc[0] for col, ser in row_match.items()}

    description = d.pop("Description")

    text = ""

    

    for k in d.keys():

        text += f"{k:22}: {d[k]}\n"

    print(text)

    print(f"< Description >\n{description}\n")



    show_pics(pet_id)
show_pet(rand_pet_id())
df_train[(df_train["Quantity"] > 10 ) & (df_train["PhotoAmt"] == 1)]
df_train[(df_train["Quantity"] == 1 ) & (df_train["Description"].str.lower().str.contains("puppies"))]
df_train[(df_train["Type"] == "Cat") & (df_train["ImageDescription"].str.contains("dog"))]