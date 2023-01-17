import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.image as mpimg

import matplotlib.lines as mlines

from tabulate import tabulate

import missingno as msno 

from IPython.display import display_html

from PIL import Image

import gc

import cv2

from scipy.stats import pearsonr,probplot, mode

import tqdm



import pydicom # for DICOM images

from skimage.transform import resize

import copy

import re



# Segmentation

from glob import glob

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scipy.ndimage

from skimage import morphology

from skimage import measure

from skimage.transform import resize

from sklearn.cluster import KMeans

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly.tools import FigureFactory as FF

from plotly.graph_objs import *

init_notebook_mode(connected=True) 



import warnings

warnings.filterwarnings("ignore")





# Set Color Palettes for the notebook

custom_colors = ['#74a09e','#86c1b2','#98e2c6','#f3c969','#f2a553', '#d96548', '#c14953']

sns.palplot(sns.color_palette(custom_colors))



# Set Style

sns.set_style("whitegrid")

sns.despine(left=True, bottom=True)
plt.rc('xtick',labelsize=11)

plt.rc('ytick',labelsize=11)
# Import train + test data

train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")



# Train len

print("Total Recordings in Train Data: {:,}".format(len(train)))
df1_styler = train.head().style.set_table_attributes("style='display:inline'").set_caption('Head Train Data')

df2_styler = test.style.set_table_attributes("style='display:inline'").set_caption('Test Data (rest Hidden)')



display_html(df1_styler._repr_html_() + df2_styler._repr_html_(), raw=True)
print("Q: Are there any missing values?", "\n" +

      "A: {}".format(train.isnull().values.any()))
print("There are {} unique patients in Train Data.".format(len(train["Patient"].unique())), "\n")



# Recordings per Patient

data = train.groupby(by="Patient")["Weeks"].count().reset_index(drop=False)

# Sort by Weeks

data = data.sort_values(['Weeks']).reset_index(drop=True)

print("Minimum number of entries are: {}".format(data["Weeks"].min()), "\n" +

      "Maximum number of entries are: {}".format(data["Weeks"].max()))



# Plot

plt.figure(figsize = (16, 6))

p = sns.barplot(data["Patient"], data["Weeks"], color=custom_colors[2])



plt.title("Number of Entries per Patient", fontsize = 17)

plt.xlabel('Patient', fontsize=14)

plt.ylabel('Frequency', fontsize=14)



p.axes.get_xaxis().set_visible(False);
# Select unique bio info for the patients

data = train.groupby(by="Patient")[["Patient", "Age", "Sex", "SmokingStatus"]].first().reset_index(drop=True)



# Figure

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (21, 7))



a = sns.distplot(data["Age"], ax=ax1, color=custom_colors[1], hist=True, kde_kws=dict(lw=0))

b = sns.countplot(data["Sex"], ax=ax2, palette=custom_colors[2:4])

c = sns.countplot(data["SmokingStatus"], ax=ax3, palette = custom_colors[4:7])



a.set_title("Patient Age Distribution", fontsize=16)

b.set_title("Sex Frequency", fontsize=16)

c.set_title("Smoking Status", fontsize=16);
print("Min FVC value: {:,}".format(train["FVC"].min()), "\n" +

      "Max FVC value: {:,}".format(train["FVC"].max()), "\n" +

      "\n" +

      "Min Percent value: {:.4}%".format(train["Percent"].min()), "\n" +

      "Max Percent value: {:.4}%".format(train["Percent"].max()))



# Figure

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (21, 7))



a = sns.distplot(train["FVC"], ax=ax1, color=custom_colors[6],kde_kws=dict(lw=3, ls="-"))

b = probplot(train['FVC'], plot=ax2)

c = sns.distplot(train["Percent"], ax=ax3, color=custom_colors[4], kde_kws=dict(lw=3, ls="-"))



a.set_title("FVC Distribution", fontsize=16)

c.set_title("Percent Distribution", fontsize=16)

plt.show()
print("Minimum no. weeks before CT: {}".format(train['Weeks'].min()), "\n" +

      "Maximum no. weeks after CT: {}".format(train['Weeks'].max()))



plt.figure(figsize = (16, 6))



a = sns.distplot(train['Weeks'], color=custom_colors[3], hist=True, kde_kws=dict(lw=3, ls="-"))

plt.title("Number of weeks before/after the CT scan", fontsize = 16)

plt.xlabel("Weeks", fontsize=14);
# Compute Correlation

corr1, _ = pearsonr(train["FVC"], train["Percent"])

corr2, _ = pearsonr(train["FVC"], train["Age"])

corr3, _ = pearsonr(train["Percent"], train["Age"])

print("Pearson Corr FVC x Percent: {:.4}".format(corr1), "\n" +

      "Pearson Corr FVC x Age: {:.0}".format(corr2), "\n" +

      "Pearson Corr Percent x Age: {:.2}".format(corr3))



# Figure

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (21, 7))



a = sns.scatterplot(x = train["FVC"], y = train["Percent"], palette=[custom_colors[2], custom_colors[6]],

                    hue = train["Sex"], style = train["Sex"], s=100, ax=ax1)



b = sns.scatterplot(x = train["FVC"], y = train["Age"], palette=[custom_colors[2], custom_colors[6]],

                    hue = train["Sex"], style = train["Sex"], s=100, ax=ax2)



c = sns.scatterplot(x = train["Percent"], y = train["Age"], palette=[custom_colors[2], custom_colors[6]],

                    hue = train["Sex"], style = train["Sex"], s=100, ax=ax3)



a.set_title("Correlation between FVC and Percent", fontsize = 16)

a.set_xlabel("FVC", fontsize = 14)

a.set_ylabel("Percent", fontsize = 14)



b.set_title("Correlation between FVC and Age", fontsize = 16)

b.set_xlabel("FVC", fontsize = 14)

b.set_ylabel("Age", fontsize = 14)



c.set_title("Correlation between Percent and Age", fontsize = 16)

c.set_xlabel("Percent", fontsize = 14)

c.set_ylabel("Age", fontsize = 14)



plt.show()
fig = plt.figure(figsize=(10, 6))



sns.barplot(x=train.groupby('Patient')['SmokingStatus'].first().value_counts().index, y=train.groupby('Patient')['SmokingStatus'].first().value_counts(), palette=custom_colors[0:3])

percentages = [(count / train.groupby('Patient')['SmokingStatus'].first().value_counts().sum() * 100).round(2) for count in train.groupby('Patient')['SmokingStatus'].first().value_counts()]



plt.ylabel('')

plt.xticks(np.arange(3), [f'Ex-smoker (%{percentages[0]})', f'Never smoked (%{percentages[1]})', f'Currently Smokes (%{percentages[2]})'])

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12)

plt.title('SmokingStatus Counts in Training Set', size=15, pad=15)



plt.show()
fig = plt.figure(figsize=(6, 6), dpi=150)



sns.heatmap(train[['Weeks', 'FVC', 'Percent', 'Age']].corr(), annot=True, square=True, cmap='summer', annot_kws={'size': 12},  fmt='.2f')   



plt.tick_params(axis='x', labelsize=11, rotation=0)

plt.tick_params(axis='y', labelsize=11, rotation=0)

plt.title('Tabular Data Features Correlations')



plt.show()
# Create base director for Train .dcm files

director = "../input/osic-pulmonary-fibrosis-progression/train"



# Create path column with the path to each patient's CT

train["Path"] = director + "/" + train["Patient"]



# Create variable that shows how many CT scans each patient has

train["CT_number"] = 0



for k, path in enumerate(train["Path"]):

    train["CT_number"][k] = len(os.listdir(path))
slice_counts = np.array([len(os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{directory}')) for directory in os.listdir('../input/osic-pulmonary-fibrosis-progression/train')])



print(f'Number of Image Slices in Training Set\n{"-" * 38}')

print(f'Mean Slice Count: {slice_counts.mean():.6}  -  Median Slice Count: {int(np.median(slice_counts))} - Total Slice Count: {slice_counts.sum()}')

print(f'Min Slice Count: {slice_counts.min()} -  Max Slice Count: {slice_counts.max()}')



fig = plt.figure(figsize=(8, 2), dpi=150)

ax = sns.countplot(slice_counts, palette='autumn')



for idx, label in enumerate(ax.get_xticklabels()):

    if idx % 10 == 0:

        label.set_visible(True)

    else:

        label.set_visible(False)



plt.ylabel('')

plt.tick_params(axis='x', labelsize=8)

plt.tick_params(axis='y', labelsize=8)

plt.title('Number of Image Slices in Training Set', size=10)



plt.show()
print("Minimum number of CT scans: {}".format(train["CT_number"].min()), "\n" +

      "Maximum number of CT scans: {:,}".format(train["CT_number"].max()))



# Scans per Patient

data = train.groupby(by="Patient")["CT_number"].first().reset_index(drop=False)

# Sort by Weeks

data = data.sort_values(['CT_number']).reset_index(drop=True)



# Plot

plt.figure(figsize = (16, 6))

p = sns.barplot(data["Patient"], data["CT_number"], color=custom_colors[5])

plt.axvline(x=85, color=custom_colors[2], linestyle='--', lw=3)



plt.title("Number of CT Scans per Patient", fontsize = 17)

plt.xlabel('Patient', fontsize=14)

plt.ylabel('Frequency', fontsize=14)



plt.text(86, 850, "Median=94", fontsize=13)



p.axes.get_xaxis().set_visible(False);
def get_metadata(patient_name):

    

    patient_directory = [pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}/{s}') for s in os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}')]

    

    try:

        patient_directory.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    except AttributeError:

        patient_directory.sort(key=lambda s: int(s.InstanceNumber))        

    

    rows = patient_directory[0].Rows

    columns = patient_directory[0].Columns

    slices = len(patient_directory)

        

    slice_thicknesses = []

    slice_spacings = []    

    rescale_slopes = []

    rescale_intercepts = []

        

    for i, s in enumerate(patient_directory):

        slice_thicknesses.append(s.SliceThickness)

        rescale_slopes.append(s.RescaleSlope)

        rescale_intercepts.append(s.RescaleIntercept)

        try:

            slice_spacings.append(s.SpacingBetweenSlices)

        except AttributeError:

            slice_spacings.append(np.nan)

                

    train.loc[train['Patient'] == patient_name, 'Rows'] = rows

    train.loc[train['Patient'] == patient_name, 'Columns'] = columns

    train.loc[train['Patient'] == patient_name, 'Slices'] = slices

    train.loc[train['Patient'] == patient_name, 'SliceThickness'] = mode(slice_thicknesses)[0][0]

    train.loc[train['Patient'] == patient_name, 'SliceSpacing'] = mode(slice_spacings)[0][0] 

    train.loc[train['Patient'] == patient_name, 'RescaleSlope'] = mode(rescale_slopes)[0][0]

    train.loc[train['Patient'] == patient_name, 'RescaleIntercept'] = mode(rescale_intercepts)[0][0]

        

for patient in tqdm.tqdm(train['Patient'].unique()):

    get_metadata(patient)

    

train['Rows'] = train['Rows'].astype(np.uint16)

train['Columns'] = train['Columns'].astype(np.uint16)

train['Slices'] = train['Slices'].astype(np.uint16)

train['SliceShape'] = train['Rows'].astype(str) + 'x' + train['Columns'].astype(str)

train['SliceThickness'] = train['SliceThickness'].astype(np.float32)

train['SliceSpacing'] = train['SliceSpacing'].astype(np.float32)

train['RescaleSlope'] = train['RescaleSlope'].astype(np.float32)

train['RescaleIntercept'] = train['RescaleIntercept'].astype(np.float32)
fig = plt.figure(figsize=(9, 3), dpi=100)

sns.barplot(x=train.groupby('Patient')['SliceShape'].first().value_counts().values, y=train.groupby('Patient')['SliceShape'].first().value_counts().index, palette='summer')



plt.xlabel('Patients', size=15)

plt.ylabel('')

plt.tick_params(axis='x', labelsize=10)

plt.tick_params(axis='y', labelsize=10)

plt.title(f'Training Set Slice Shapes', size=15)



plt.show()
def crop_slice(s):



    """

    Crop frames from slices



    Parameters

    ----------

    s : numpy array, shape = (Rows, Columns)

    numpy array of slices with frame



    Returns

    -------

    s_cropped : numpy array, shape = (Rows - All Zero Rows, Columns - All Zero Columns)

    numpy array after the all zero rows and columns are dropped

    """



    s_cropped = s[~np.all(s == 0, axis=1)]

    s_cropped = s_cropped[:, ~np.all(s_cropped == 0, axis=0)]

    return s_cropped
fig, axes = plt.subplots(figsize=(20, 9), nrows=2, ncols=5)



for i, patient_name in enumerate(train.groupby('SliceShape')['Patient'].first()):   

    

    patient_directory = [pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}/{s}') for s in os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{patient_name}')]

    patient_directory.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    first_slice_cropped = crop_slice(patient_directory[0].pixel_array)

    if first_slice_cropped.shape[0] != 512 and first_slice_cropped.shape[1] != 512:

        first_slice_cropped_resized = cv2.resize(first_slice_cropped, (512, 512), interpolation=cv2.INTER_NEAREST)

    else:

        first_slice_cropped_resized = first_slice_cropped

    old_aspect_ratio = train.groupby("SliceShape")["Patient"].first().index[i]

    new_aspect_ratio = f'{first_slice_cropped_resized.shape[0]}x{first_slice_cropped_resized.shape[1]}'

    

    if i > 4:

        i -= 5

        j = 1

    else:

        j = 0

        

    axes[j][i].imshow(first_slice_cropped_resized, cmap=plt.cm.bone)



    axes[j][i].tick_params(axis='x', labelsize=14)

    axes[j][i].tick_params(axis='y', labelsize=14)

    axes[j][i].set_title(f'{old_aspect_ratio} -> {new_aspect_ratio}', size=14)



plt.show()
# https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/



def make_lungmask(img, display=False):

    row_size= img.shape[0]

    col_size = img.shape[1]

    

    mean = np.mean(img)

    std = np.std(img)

    img = img-mean

    img = img/std

    

    # Find the average pixel value near the lungs

        # to renormalize washed out images

    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 

    mean = np.mean(middle)  

    max = np.max(img)

    min = np.min(img)

    

    # To improve threshold finding, I'm moving the 

    # underflow and overflow on the pixel spectrum

    img[img==max]=mean

    img[img==min]=mean

    

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)

    

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))

    centers = sorted(kmeans.cluster_centers_.flatten())

    threshold = np.mean(centers)

    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image



    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  

    # We don't want to accidentally clip the lung.



    eroded = morphology.erosion(thresh_img,np.ones([3,3]))

    dilation = morphology.dilation(eroded,np.ones([8,8]))



    labels = measure.label(dilation) # Different labels are displayed in different colors

    label_vals = np.unique(labels)

    regions = measure.regionprops(labels)

    good_labels = []

    for prop in regions:

        B = prop.bbox

        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:

            good_labels.append(prop.label)

    mask = np.ndarray([row_size,col_size],dtype=np.int8)

    mask[:] = 0





    #  After just the lungs are left, we do another large dilation

    #  in order to fill in and out the lung mask 

    

    for N in good_labels:

        mask = mask + np.where(labels==N,1,0)

    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation



    if (display):

        fig, ax = plt.subplots(3, 2, figsize=[12, 12])

        ax[0, 0].set_title("Original")

        ax[0, 0].imshow(img, cmap='gray')

        ax[0, 0].axis('off')

        ax[0, 1].set_title("Threshold")

        ax[0, 1].imshow(thresh_img, cmap='gray')

        ax[0, 1].axis('off')

        ax[1, 0].set_title("After Erosion and Dilation")

        ax[1, 0].imshow(dilation, cmap='gray')

        ax[1, 0].axis('off')

        ax[1, 1].set_title("Color Labels")

        ax[1, 1].imshow(labels)

        ax[1, 1].axis('off')

        ax[2, 0].set_title("Final Mask")

        ax[2, 0].imshow(mask, cmap='gray')

        ax[2, 0].axis('off')

        ax[2, 1].set_title("Apply Mask on Original")

        ax[2, 1].imshow(mask*img, cmap='gray')

        ax[2, 1].axis('off')

        

        plt.show()

    return mask*img
# Select a sample

path = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/19.dcm"

dataset = pydicom.dcmread(path)

img = dataset.pixel_array



# Masked image

mask_img = make_lungmask(img, display=True)
patient_dir = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430"

datasets = []



# First Order the files in the dataset

files = []

for dcm in list(os.listdir(patient_dir)):

    files.append(dcm) 

files.sort(key=lambda f: int(re.sub('\D', '', f)))



# Read in the Dataset

for dcm in files:

    path = patient_dir + "/" + dcm

    datasets.append(pydicom.dcmread(path))

    

imgs = []

for data in datasets:

    img = data.pixel_array

    imgs.append(img)

    

    

# Show masks

fig=plt.figure(figsize=(16, 6))

columns = 10

rows = 3



for i in range(1, columns*rows +1):

    img = make_lungmask(datasets[i-1].pixel_array)

    fig.add_subplot(rows, columns, i)

    plt.imshow(img, cmap="gray")

    plt.title(i, fontsize = 9)

    plt.axis('off');