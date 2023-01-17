# Regular Imports

import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.image as mpimg

from tabulate import tabulate

import missingno as msno 

from IPython.display import display_html

from PIL import Image

import gc

import cv2

from scipy.stats import pearsonr

import tqdm



import pydicom # for DICOM images

from skimage.transform import resize

import copy

import re



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



# Color Palette

custom_colors = ['#6C4C4D','#95715F','#C4A797','#A9DBC2','#3C887E', '#386B64']

sns.palplot(sns.color_palette(custom_colors))



# Set Style

sns.set_style("whitegrid")

sns.despine(left=True, bottom=True)



# Set tick size

plt.rc('xtick',labelsize=11)

plt.rc('ytick',labelsize=11)
# Centers the images



from IPython.core.display import HTML

HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

</style>

""")
basepath = "../input/rsna-str-pulmonary-embolism-detection"



# Import the data

train_csv = pd.read_csv(basepath + "/train.csv")

test_csv = pd.read_csv(basepath + "/test.csv")



# How many lines are in each dataframe?

print("Train: {:,} lines.".format(len(train_csv)), "\n" +

      "Test: {:,} lines.".format(len(test_csv)))
df1_styler = train_csv.head().style.set_table_attributes("style='display:inline'").set_caption('Head Train Data')

df2_styler = test_csv.head().style.set_table_attributes("style='display:inline'").set_caption('Test Data (rest Hidden)')



display_html(df1_styler._repr_html_() + df2_styler._repr_html_(), raw=True)
print("Train Data:", "\n" + 

      "Q: Are there any missing values?", "\n" +

      "A: {}".format(train_csv.isnull().values.any()), "\n")



print("Test Data:", "\n" + 

      "Q: Are there any missing values?", "\n" +

      "A: {}".format(test_csv.isnull().values.any()))
# Number of unique studies

print("--- Train:", "\n" +

      "Total Unique Studies&Series: {:,}".format(len(train_csv.groupby("StudyInstanceUID")["SeriesInstanceUID"].count())))



# Group by Study

data = train_csv.groupby("StudyInstanceUID")["SOPInstanceUID"].count().reset_index()

print("Min No. Images/Study: {:,}".format(data.min()[1]), "\n" +

      "Max No. Images/Study: {:,}".format(data.max()[1]), "\n" +

      "Avg No. Images/Study: {:,.0f}".format(round(data.mean()[0], 0)))



# Plot

plt.figure(figsize=(16, 6))

sns.boxenplot(x = "SOPInstanceUID", data=data, color=custom_colors[4])





plt.title("TRAIN: No. Images per Study", fontsize = 17)

plt.xlabel('No. Images', fontsize=14);
# Number of unique studies

print("--- Test:", "\n" +

      "Total Unique Studies&Series: {:,}".format(len(test_csv.groupby("StudyInstanceUID")["SeriesInstanceUID"].count())))



# Group by Study

data = test_csv.groupby("StudyInstanceUID")["SOPInstanceUID"].count().reset_index()

print("Min No. Images/Study: {:,}".format(data.min()[1]), "\n" +

      "Max No. Images/Study: {:,}".format(data.max()[1]), "\n" +

      "Avg No. Images/Study: {:,.0f}".format(round(data.mean()[0], 0)))



# Plot

plt.figure(figsize=(16, 6))

sns.boxenplot(x = "SOPInstanceUID", data=data, color=custom_colors[2])





plt.title("TEST: No. Images per Study", fontsize = 17)

plt.xlabel('No. Images', fontsize=14);
predict_variables = ['pe_present_on_image', 'negative_exam_for_pe', 'rv_lv_ratio_gte_1', 

                     'rv_lv_ratio_lt_1', 'leftsided_pe', 'chronic_pe', 'rightsided_pe', 

                     'acute_and_chronic_pe', 'central_pe', 'indeterminate']



# Melt the prediction variables on a single column - so we can plot more easily the 10 variables

melt = train_csv[['SOPInstanceUID'] + predict_variables]

melt = pd.melt(melt, id_vars=['SOPInstanceUID'], value_vars=predict_variables)
plt.figure(figsize=(16, 6))



a = sns.countplot(x=melt["variable"], hue=melt["value"], 

                  palette="copper")



plt.xticks(rotation=40)

plt.title("TRAIN: (to predict) Variables", fontsize = 16)

plt.xlabel("")

plt.ylabel("Frequency", fontsize=14);
bonus_variables = ['qa_motion', 'qa_contrast', 'flow_artifact', 'true_filling_defect_not_pe']



# Melt the variables on a single column - so we can plot more easily the 4 variables

melt = train_csv[['SOPInstanceUID'] + bonus_variables]

melt = pd.melt(melt, id_vars=['SOPInstanceUID'], value_vars=bonus_variables)
plt.figure(figsize=(16, 6))



a = sns.countplot(x=melt["variable"], hue=melt["value"], 

                  palette="PuBuGn_r")



plt.xticks(rotation=40)

plt.title("TRAIN: (bonus) Variables", fontsize = 16)

plt.xlabel("")

plt.ylabel("Frequency", fontsize=14);
# Create base director for train and test data

base_train = "../input/rsna-str-pulmonary-embolism-detection/train"

base_test = "../input/rsna-str-pulmonary-embolism-detection/test"



# --- TRAIN

# Count total number of files in each subdirectory in train and test

dcm_train = 0



# dirpath - the directory path in string

# dirnames - all main directories

# filenames - all subdirectories

for dirpath, dirnames, filenames in tqdm.tqdm(os.walk(base_train)):

    dcm_train += len(filenames)

        

# --- TEST

dcm_test = 0



for dirpath, dirnames, filenames in tqdm.tqdm(os.walk(base_test)):

    dcm_test += len(filenames)
print("Train: total .dcm files - {:,}".format(dcm_train), "\n" +

      "Test: total .dcm files - {:,}".format(dcm_test))
# Color of text

class bcolors:

    OKBLUE = '\033[96m'

    OKGREEN = '\033[92m'

    

path = "../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf/5eb3f6566b0f.dcm"

dataset = pydicom.dcmread(path)



print(bcolors.OKBLUE + "Image Type.......:", dataset.ImageType, "\n" +

      "Modality.........:", dataset.Modality, "\n" +

      "Rows.............:", dataset.Rows, "\n" +

      "Columns..........:", dataset.Columns)



plt.figure(figsize = (7, 7))

plt.imshow(dataset.pixel_array, cmap="plasma")

plt.axis('off');
# Get the base directory

base = "../input/rsna-str-pulmonary-embolism-detection/train"



# Get a sample of 2 images (1 with pe present and 1 with no pe present)

pe_yes = train_csv[train_csv["pe_present_on_image"] == 1].sample(random_state=33).reset_index()

pe_no = train_csv[train_csv["pe_present_on_image"] == 0].sample(random_state=101).reset_index()



# Get paths of these images

pe_yes_path = base + "/" + pe_yes["StudyInstanceUID"] + "/" + pe_yes["SeriesInstanceUID"] + "/" + pe_yes["SOPInstanceUID"] + ".dcm"

pe_no_path = base + "/" + pe_no["StudyInstanceUID"] + "/" + pe_no["SeriesInstanceUID"] + "/" + pe_no["SOPInstanceUID"] + ".dcm"



pe_yes_path = pe_yes_path[0]

pe_no_path = pe_no_path[0]
pe_yes[["SOPInstanceUID", "pe_present_on_image", "leftsided_pe", "rightsided_pe", "central_pe"]]
dataset_yes = pydicom.dcmread(pe_yes_path)

dataset_no = pydicom.dcmread(pe_no_path)





f, ax = plt.subplots(1, 2, figsize = (11, 11))



ax[0].imshow(dataset_no.pixel_array, cmap="plasma")

ax[1].imshow(dataset_yes.pixel_array, cmap="plasma")

ax[0].title.set_text("No PE present")

ax[1].title.set_text("PE present")

ax[0].axis('off')

ax[1].axis('off');
# Study "0003b3d648eb"

study_dir = "../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf"

datasets = []



# Read in the Dataset

for dcm in os.listdir(study_dir):

    path = study_dir + "/" + dcm

    datasets.append(pydicom.dcmread(path))
# Plot the images

fig=plt.figure(figsize=(16, 6))

columns = 10

rows = 3



for i in range(1, columns*rows +1):

    img = datasets[i-1].pixel_array

    fig.add_subplot(rows, columns, i)

    plt.imshow(img, cmap="plasma")

    plt.title(i, fontsize = 9)

    plt.axis('off');