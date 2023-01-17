
from IPython.display import IFrame, YouTubeVideo
YouTubeVideo('N2apCx1rlIQ',width=600, height=400)
YouTubeVideo('3fNf8KX1AlQ',width=600, height=400)
!pip install joypy --progress-bar off
import os


import random
import seaborn as sns
import cv2
# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL

import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib
import h5py
import plotly.graph_objs as go
from IPython.display import Image, display
import joypy
import warnings
warnings.filterwarnings("ignore")
os.listdir('/kaggle/input/trends-assessment-prediction/')
BASE_PATH = '../input/trends-assessment-prediction'

# image and mask directories
train_data_dir = f'{BASE_PATH}/fMRI_train'
test_data_dir = f'{BASE_PATH}/fMRI_test'


print('Reading data...')
loading_data = pd.read_csv(f'{BASE_PATH}/loading.csv')
train_data = pd.read_csv(f'{BASE_PATH}/train_scores.csv')
sample_submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
print('Reading data completed')
display(train_data.head())
print("Shape of train_data :", train_data.shape)
display(loading_data.head())
print("Shape of loading_data :", loading_data.shape)
# checking missing data
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()
total = loading_data.isnull().sum().sort_values(ascending = False)
percent = (loading_data.isnull().sum()/loading_data.isnull().count()*100).sort_values(ascending = False)
missing_loading_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_loading_data.head()
def plot_bar(df, feature, title='', show_percent = False, size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.barplot(np.round(df[feature].value_counts().index).astype(int), df[feature].value_counts().values, alpha=0.8, palette='Set2')

    plt.title(title)
    if show_percent:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(100*height/total),
                    ha="center", rotation=45) 
    plt.xlabel(feature, fontsize=12, )
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()
plot_bar(train_data, 'age', 'age count and %age plot', show_percent=True, size=4)
def plot_bar(df, feature, title='', show_percent = False, size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.barplot(np.round(df[feature].value_counts().index).astype(int), df[feature].value_counts().values, alpha=0.8, palette='Set2')

    plt.title(title)
    if show_percent:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(100*height/total),
                    ha="center", rotation=45) 
    plt.xlabel(feature, fontsize=12, )
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()
### Age count Distribution
for col in train_data.columns[2:]:
    plot_bar(train_data, col, f'{col} count plot', size=4)
temp_data =  train_data.drop(['Id'], axis=1)

plt.figure(figsize = (12, 8))
sns.heatmap(temp_data.corr(), annot = True, cmap="RdYlGn")
plt.yticks(rotation=0) 

plt.show()
temp_data =  loading_data.drop(['Id'], axis=1)

plt.figure(figsize = (20, 20))
sns.heatmap(temp_data.corr(), annot = True, cmap="RdYlGn")
plt.yticks(rotation=0) 

plt.show()
temp_data =  loading_data.drop(['Id'], axis=1)
# Create correlation matrix
correl = temp_data.corr().abs()

# Select upper triangle of correlation matrix
upper = correl.where(np.triu(np.ones(correl.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.5
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]

print('Very high correlated features: ', to_drop)
# Draw Plot
import joypy

targets= loading_data.columns[1:]


plt.figure(figsize=(16,10), dpi= 80)
fig, axes = joypy.joyplot(loading_data, column=list(targets), ylim='own', figsize=(14,10))

# Decoration
plt.title('Distribution of features IC_01 to IC_29', fontsize=22)
plt.show()
# Download the ch2better template image for display
!wget https://github.com/Chaogan-Yan/DPABI/raw/master/Templates/ch2better.nii
"""
    Load and display a subject's spatial map
"""

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
mask_filename = f'{BASE_PATH}/fMRI_mask.nii'
subject_filename = '../input/trends-assessment-prediction/fMRI_train/10004.mat'
smri_filename = 'ch2better.nii'
mask_niimg = nl.image.load_img(mask_filename)


def load_subject(filename, mask_niimg):
    """
    Load a subject saved in .mat format with
        the version 7.3 flag. Return the subject
        niimg, using a mask niimg as a template
        for nifti headers.
        
    Args:
        filename    <str>            the .mat filename for the subject data
        mask_niimg  niimg object     the mask niimg object used for nifti headers
    """
    subject_data = None
    with h5py.File(subject_filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])
    subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
    return subject_niimg
subject_niimg = load_subject(subject_filename, mask_niimg)
print("Image shape is %s" % (str(subject_niimg.shape)))
num_components = subject_niimg.shape[-1]
print("Detected {num_components} spatial maps".format(num_components=num_components))
nlplt.plot_prob_atlas(subject_niimg, bg_img=smri_filename, view_type='filled_contours', draw_cross=False,title='All %d spatial maps' % num_components, threshold='auto')
grid_size = int(np.ceil(np.sqrt(num_components)))
fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*10, grid_size*10))
[axi.set_axis_off() for axi in axes.ravel()]
row = -1
for i, cur_img in enumerate(nl.image.iter_img(subject_niimg)):
    col = i % grid_size
    if col == 0:
        row += 1
    nlplt.plot_stat_map(cur_img, bg_img=smri_filename, title="IC %d" % i, axes=axes[row, col], threshold=3, colorbar=False)
import warnings
warnings.filterwarnings("ignore")

!pip install pycaret --quiet
from pycaret.regression import *

BASE_PATH = '../input/trends-assessment-prediction'

fnc_df = pd.read_csv(f"{BASE_PATH}/fnc.csv")
loading_df = pd.read_csv(f"{BASE_PATH}/loading.csv")
labels_df = pd.read_csv(f"{BASE_PATH}/train_scores.csv")
fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])
df = fnc_df.merge(loading_df, on="Id")
labels_df["is_train"] = True
df = df.merge(labels_df, on="Id", how="left")

test_df = df[df["is_train"] != True].copy()

target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
test_df = test_df.drop(target_cols + ['is_train'], axis=1)

# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.
FNC_SCALE = 1/500
test_df[fnc_features] *= FNC_SCALE
target_models_dict = {
    'age': 'age_br',
    'domain1_var1':'domain1_var1_ridge',
    'domain1_var2':'domain1_var2_svm',
    'domain2_var1':'domain2_var1_ridge',
    'domain2_var2':'domain2_var2_svm',
}
## load PyCaret models

for index, target in enumerate(target_cols):
    model_name = target_models_dict[target]
    model = load_model(f'../input/pycaret-trends-models/{model_name}', platform = None, authentication = None, verbose=True)

    predictions = predict_model(model, data=test_df)
    test_df[target] = predictions['Label'].values
sub_df = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis=1).sort_values("Id")
assert sub_df.shape[0] == test_df.shape[0]*5

sub_df.to_csv("submission1.csv", index=False)
sub_df.head()