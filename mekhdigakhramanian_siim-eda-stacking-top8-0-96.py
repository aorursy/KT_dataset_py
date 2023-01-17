import pandas as pd

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import io

import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go#visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization



import seaborn as sns

import plotly.express as px



import pydicom # for DICOM images

from skimage.transform import resize



# SKLearn

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



import os

import random

import re

import math

import time

from IPython.display import display_html

import missingno as msno 

import gc

import cv2

import matplotlib.image as mpimg



# Set Color Palettes for the notebook (https://color.adobe.com/)

colors_nude = ['#FFE61A','#B2125F','#FF007B','#14B4CC','#099CB3']

sns.palplot(sns.color_palette(colors_nude))



# Set Style

sns.set_style("whitegrid")

sns.despine(left=True, bottom=True)





import warnings



warnings.filterwarnings('ignore') # Disabling warnings for clearer outputs





seed_val = 42

random.seed(seed_val)

np.random.seed(seed_val)
# loading datasets



train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

sample = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
train.sample(5)
test.sample(5)
# loading datasets



train = pd.read_csv('../input/melanomaextendedtabular/external_upsampled_tabular.csv')

test = pd.read_csv('../input/melanomaextendedtabular/test_tabular.csv')

sample = pd.read_csv('../input/melanomaextendedtabular/sample_submission.csv')
# checking column names



print(

    f'Train data has {train.shape[1]} features, {train.shape[0]} observations and Test data {test.shape[1]} features, {test.shape[0]} observations.\nTrain features are:\n{train.columns.tolist()}\nTest features are:\n{test.columns.tolist()}'

)
# renaming column names for easier use



train.columns = [

    'img_name','sex', 'age', 'location', 'target','width','height'

]



test.columns = ['img_name','sex', 'age', 'location','width', 'height']
# Checking missing values:



def missing_percentage(df):



    total = df.isnull().sum().sort_values(

        ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]

    percent = (df.isnull().sum().sort_values(ascending=False) / len(df) *

               100)[(df.isnull().sum().sort_values(ascending=False) / len(df) *

                     100) != 0]

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])





missing_train = missing_percentage(train)

missing_test = missing_percentage(test)



fig, ax = plt.subplots(1, 2, figsize=(16, 6))



sns.barplot(x=missing_train.index,

            y='Percent',

            data=missing_train,

            palette=colors_nude,

            ax=ax[0])



sns.barplot(x=missing_test.index,

            y='Percent',

            data=missing_test,

            palette=colors_nude,

            ax=ax[1])



ax[0].set_title('Train Data Missing Values')

ax[1].set_title('Test Data Missing Values')
# Directory

directory = '../input/siim-isic-melanoma-classification'



# Import the 2 csv s

train_df = pd.read_csv(directory + '/train.csv')

test_df = pd.read_csv(directory + '/test.csv')



print('Train has {:,} rows and Test has {:,} rows.'.format(len(train_df), len(test_df)))



# Change columns names

new_names = ['dcm_name', 'ID', 'sex', 'age', 'anatomy', 'diagnosis', 'benign_malignant', 'target']

train_df.columns = new_names

test_df.columns = new_names[:5]
df1_styler = train_df.head().style.set_table_attributes("style='display:inline'").set_caption('Head Train Data')

df2_styler = test_df.head().style.set_table_attributes("style='display:inline'").set_caption('Head Test Data')



display_html(df1_styler._repr_html_() + df2_styler._repr_html_(), raw=True)
f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



msno.matrix(train_df, ax = ax1, color=(107/255, 196/255, 171/255), fontsize=10)

msno.matrix(test_df, ax = ax2, color=(136/255, 136/255, 130/255), fontsize=10)



ax1.set_title('Train Missing Values', fontsize = 16)

ax2.set_title('Test Missing Values', fontsize = 16);
#labels

lab = train["sex"].value_counts().keys().tolist()



#values

val = train["sex"].value_counts().values.tolist()



trace1 = go.Pie(labels = lab ,

               values = val ,

               marker = dict(colors =  [ 'royalblue' ,'lime'],

                             line = dict(color = "white",

                                         width =  1.3)

                            ),

               rotation = 90,

               hoverinfo = "label+value+text",

               hole = .5

              )

layout = go.Layout(dict(title = "Gender Distribution",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                       )

                  )





lab2 = train["age"].value_counts().keys().tolist()



#values

val2 = train["age"].value_counts().values.tolist()



trace2 = go.Pie(labels = lab2 ,

               values = val2 ,

               marker = dict(colors =  [ 'royalblue' ,'lime'],

                             line = dict(color = "white",

                                         width =  1.3)

                            ),

               rotation = 90,

               hoverinfo = "label+value+text",

               hole = .5

              )

layout = go.Layout(dict(title = "Age Distribution",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                       )

                  )



data = [trace1,trace2]

fig  = go.Figure(data = data,layout = layout)

py.iplot(fig)

# Filling missing  values with 'unknown' and '-1' tags:



for df in [train, test]:

    df['location'].fillna('unknown', inplace=True)

    

train['sex'].fillna('unknown', inplace=True)



train['age'].fillna(-1, inplace=True)
# Plotting interactive sunburst:



fig = px.sunburst(data_frame=train,

                  path=['target', 'sex', 'location'],

                  color='sex',

                  color_discrete_sequence=colors_nude,

                  maxdepth=-1,

                  title='Sunburst Chart Benign/Malignant > Sex > Location')



fig.update_traces(textinfo='label+percent parent')

fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))

fig.show()
# Impute for anatomy

train_df['anatomy'].fillna('torso', inplace = True) 
anatomy = test_df.copy()

anatomy['flag'] = np.where(test_df['anatomy'].isna()==True, 'missing', 'not_missing')



# Figure

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



sns.countplot(anatomy['flag'], hue=anatomy['sex'], ax=ax1, palette=colors_nude)



sns.distplot(anatomy[anatomy['flag'] == 'missing']['age'],

             hist=False, rug=True, label='Missing', ax=ax2, 

             color=colors_nude[2], kde_kws=dict(linewidth=4, bw=0.1))



sns.distplot(anatomy[anatomy['flag'] == 'not_missing']['age'], 

             hist=False, rug=True, label='Not Missing', ax=ax2, 

             color=colors_nude[3], kde_kws=dict(linewidth=4, bw=0.1))



ax1.set_title('Gender for Anatomy', fontsize=16)

ax2.set_title('Age Distribution for Anatomy', fontsize=16)

sns.despine(left=True, bottom=True);
# Figure

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6))



a = sns.countplot(train_df['anatomy'], ax=ax1, palette = colors_nude)

b = sns.countplot(train_df['diagnosis'], ax=ax2, palette = colors_nude)



a.set_xticklabels(a.get_xticklabels(), rotation=35, ha="right")

b.set_xticklabels(b.get_xticklabels(), rotation=35, ha="right")



for p in a.patches:

    a.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')

    

for p in b.patches:

    b.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')

    

ax1.set_title('Anatomy Frequencies', fontsize=16)

ax2.set_title('Diagnosis Frequencies', fontsize=16)

sns.despine(left=True, bottom=True);
print(

    f'Number of unique Patient ID\'s in train set: {train_df.nunique()}, Total: {train_df.count()}\nNumber of unique Patient ID\'s in test set: {test_df.nunique()}, Total: {test_df.count()}'

)
print('Train .dcm number of images:', len(list(os.listdir('../input/siim-isic-melanoma-classification/train'))), '\n' +

      'Test .dcm number of images:', len(list(os.listdir('../input/siim-isic-melanoma-classification/test'))), '\n' +

      'Train .jpeg number of images:', len(list(os.listdir('../input/siim-isic-melanoma-classification/jpeg/train'))), '\n' +

      'Test .jpeg number of images:', len(list(os.listdir('../input/siim-isic-melanoma-classification/jpeg/test'))), '\n' +

      '-----------------------', '\n' +

      'There is the same number of images as in train/ test .csv datasets')
# Add Image Path



# === DICOM ===

# Create the paths

path_train = directory + '/train/' + train_df['dcm_name'] + '.dcm'

path_test = directory + '/test/' + test_df['dcm_name'] + '.dcm'



# Append to the original dataframes

train_df['path_dicom'] = path_train

test_df['path_dicom'] = path_test



# === JPEG ===

# Create the paths

path_train = directory + '/jpeg/train/' + train_df['dcm_name'] + '.jpg'

path_test = directory + '/jpeg/test/' + test_df['dcm_name'] + '.jpg'



# Append to the original dataframes

train_df['path_jpeg'] = path_train

test_df['path_jpeg'] = path_test

# Save the files

train_df.to_csv('train_clean.csv', index=False)

test_df.to_csv('test_clean.csv', index=False)
# === DICOM ===

# Create the paths

path_train = directory + '/train/' + train_df['dcm_name'] + '.dcm'

path_test = directory + '/test/' + test_df['dcm_name'] + '.dcm'



# Append to the original dataframes

train_df['path_dicom'] = path_train

test_df['path_dicom'] = path_test



# === JPEG ===

# Create the paths

path_train = directory + '/jpeg/train/' + train_df['dcm_name'] + '.jpg'

path_test = directory + '/jpeg/test/' + test_df['dcm_name'] + '.jpg'



# Append to the original dataframes

train_df['path_jpeg'] = path_train

test_df['path_jpeg'] = path_test
def show_images(data, n = 5, rows=1, cols=5, title='Default'):

    plt.figure(figsize=(16,4))



    for k, path in enumerate(data['path_dicom'][:n]):

        image = pydicom.read_file(path)

        image = image.pixel_array

        

        # image = resize(image, (200, 200), anti_aliasing=True)



        plt.suptitle(title, fontsize = 16)

        plt.subplot(rows, cols, k+1)

        plt.imshow(image)

        plt.axis('off')
# Show Benign Samples

show_images(train_df[train_df['target'] == 0], n=10, rows=2, cols=5, title='Benign Sample')
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(16,6))

plt.suptitle("B&W", fontsize = 16)



for i in range(0, 2*6):

    data = pydicom.read_file(train_df['path_dicom'][i])

    image = data.pixel_array

    

    # Transform to B&W

    # The function converts an input image from one color space to another.

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (200,200))

    

    x = i // 6

    y = i % 6

    axes[x, y].imshow(image, cmap=plt.cm.bone) 

    axes[x, y].axis('off')
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(16,6))

plt.suptitle("Without Gaussian Blur", fontsize = 16)



for i in range(0, 2*6):

    data = pydicom.read_file(train_df['path_dicom'][i])

    image = data.pixel_array

    

    # Transform to B&W

    # The function converts an input image from one color space to another.

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    image = cv2.resize(image, (200,200))

    

    x = i // 6

    y = i % 6

    axes[x, y].imshow(image, cmap=plt.cm.bone) 

    axes[x, y].axis('off')
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(16,6))

plt.suptitle("With Gaussian Blur", fontsize = 16)



for i in range(0, 2*6):

    data = pydicom.read_file(train_df['path_dicom'][i])

    image = data.pixel_array

    

    # Transform to B&W

    # The function converts an input image from one color space to another.

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    image = cv2.resize(image, (200,200))

    image=cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0) ,256/10), -4, 128)

    

    x = i // 6

    y = i % 6

    axes[x, y].imshow(image, cmap=plt.cm.bone) 

    axes[x, y].axis('off')
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(16,6))

plt.suptitle("Hue, Saturation, Brightness", fontsize = 16)



for i in range(0, 2*6):

    data = pydicom.read_file(train_df['path_dicom'][i])

    image = data.pixel_array

    

    # Transform to B&W

    # The function converts an input image from one color space to another.

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    image = cv2.resize(image, (200,200))

    

    x = i // 6

    y = i % 6

    axes[x, y].imshow(image, cmap=plt.cm.bone) 

    axes[x, y].axis('off')
# Necessary Imports

import torch

from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms

import torchvision
# Select a small sample of the .jpeg image paths

image_list = train_df.sample(12)['path_jpeg']

image_list = image_list.reset_index()['path_jpeg']



# Show the sample

plt.figure(figsize=(16,6))

plt.suptitle("Original View", fontsize = 16)

    

for k, path in enumerate(image_list):

    image = mpimg.imread(path)

        

    plt.subplot(2, 6, k+1)

    plt.imshow(image)

    plt.axis('off')
WEIGHT = 1
submission = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

sub_best = pd.read_csv('../input/eda-modelling-of-the-external-data-inc-ensemble/external_meta_ensembled.csv')
files_sub = [

    '../input/minmax-ensemble-0-9526-lb/submission.csv',

    '../input/new-basline-np-log2-ensemble-top-10/submission.csv',

    '../input/stacking-ensemble-on-my-submissions/submission_mean.csv',

    '../input/analysis-of-melanoma-metadata-and-effnet-ensemble/ensembled.csv',

    '../input/eda-modelling-of-the-external-data-inc-ensemble/external_meta_ensembled.csv',

    '../input/submission-exploration/submission.csv',

    '../input/siim-isic-melanoma-classification-ensemble/submission.csv',

    '../input/stacking-ensemble-on-my-submissions/submission_median.csv',

    '../input/analysis-of-melanoma-metadata-and-effnet-ensemble/blended_effnets.csv'

]

files_sub = sorted(files_sub)

print(len(files_sub))

files_sub
for file in files_sub:

    test[file.replace(".csv", "")] = pd.read_csv(file).sort_values('image_name')["target"]

test['id'] = test.index
test.head()
test.columns
test["diff_good1"] =  test['../input/new-basline-np-log2-ensemble-top-10/submission'] - test['../input/stacking-ensemble-on-my-submissions/submission_mean']

test["diff_good1"] =  test['../input/eda-modelling-of-the-external-data-inc-ensemble/external_meta_ensembled'] - test['../input/siim-isic-melanoma-classification-ensemble/submission']

test["diff_good2"] = test['../input/analysis-of-melanoma-metadata-and-effnet-ensemble/blended_effnets'] - test['../input/stacking-ensemble-on-my-submissions/submission_median']



test["diff_bad1"] = test['../input/minmax-ensemble-0-9526-lb/submission'] - test['../input/submission-exploration/submission']
test["sub_best"] = test['../input/analysis-of-melanoma-metadata-and-effnet-ensemble/ensembled']

col_comment = ["id", "image_name", "patient_id", "sub_best"]

col_diff = [column for column in test.columns if "diff" in column]

test_diff = test[col_comment + col_diff].reset_index(drop=True)



test_diff["diff_avg"] = test_diff[col_diff].mean(axis=1) # the mean trend
# Apply the post-processing technique in one line (as explained in the pseudo-code of my post.

test_diff["sub_new"] = test_diff.apply(lambda x: (1+WEIGHT*x["diff_avg"])*x["sub_best"] if x["diff_avg"]<0 else (1-WEIGHT*x["diff_avg"])*x["sub_best"] + WEIGHT*x["diff_avg"] , axis=1)
submission["target"] = sub_best["target"]

submission.head()
test_diff.head()
submission.loc[test["id"], "target"] = test_diff["sub_new"].values
submission.to_csv("submission.csv", index=False)

submission.head()
plt.hist(submission.target,bins=100)

plt.show()