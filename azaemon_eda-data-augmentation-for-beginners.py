import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
import matplotlib.image as mpimg

# Set Color Palettes for the notebook (https://color.adobe.com/)
colors_nude = ['#FFE61A','#B2125F','#FF007B','#14B4CC','#099CB3']
sns.palplot(sns.color_palette(colors_nude))

# Set Style
sns.set_style("whitegrid")
sns.despine(left=True, bottom=True)
train_data= pd.read_csv("../input/landmark-recognition-2020/train.csv")
print(train_data.head())
print()
print("Here, id means Image Id\n      landmark_id points to a specific ID of the landmark ")
train_data.describe()
print(train_data.isna().sum())
print()
print('Here, we can see there is no missing data in any of the columns.')
!pip install basic_image_eda
from basic_image_eda import BasicImageEDA
data_dir = "../input/landmark-recognition-2020/train/0"
extensions = ['png', 'jpg', 'jpeg']
threads = 0
dimension_plot = True
channel_hist = True
nonzero = False
hw_division_factor = 1.0

BasicImageEDA.explore(data_dir, extensions, threads, dimension_plot, channel_hist, nonzero, hw_division_factor)
train_data['landmark_id'].value_counts()
print("Types of Landmarks: 81313")
print("Landmark ID: 138982 has the highest number of images (6272)")
# Occurance of landmark_id in decreasing order(Top categories)
temp = pd.DataFrame(train_data.landmark_id.value_counts().head(10))
temp.reset_index(inplace=True)
temp.columns = ['Landmark ID','Number of Images']

# Plot the most frequent landmark_ids
plt.figure(figsize = (9, 10))
plt.title('Top 10 the mostfrequent landmarks')
sns.set_color_codes("deep")
sns.barplot(x="Landmark ID", y="Number of Images", data=temp,
            label="Count")
plt.show()

temp = pd.DataFrame(train_data.landmark_id.value_counts().tail(10))
temp.reset_index(inplace=True)
temp.columns = ['Landmark ID','Number of Images']
# Plot the least frequent landmark_ids
plt.figure(figsize = (9, 10))
plt.title('Top 10 the least frequent landmarks')
sns.set_color_codes("deep")
sns.barplot(x="Landmark ID", y="Number of Images", data=temp,
            label="Count")
plt.show()

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
train_data['landmark_id'].value_counts(normalize=True).sort_values().iplot(kind='barh',
                                                      xTitle='Percentage', 
                                                      linecolor='black', 
                                                      opacity=0.7,
                                                      color='blue',
                                                      theme='pearl',
                                                      bargap=0.2,
                                                      gridcolor='white',
                                                      title='Distribution in the training set')
sns.distplot(temp['Landmark ID'], hist=True, rug=True);
from random import randrange
fig= plt.figure(figsize=(20,10))
index= '../input/landmark-recognition-2020/train/7/0/4/704001a0be55059a.jpg'
a= fig.add_subplot(2,3,1)
a.set_title(index.split("/")[-1])
plt.imshow(plt.imread(index))

index= '../input/landmark-recognition-2020/train/7/0/4/7040a5cfa43e0633.jpg'
a= fig.add_subplot(2,3,2)
a.set_title(index.split("/")[-1])
plt.imshow(plt.imread(index))

index= '../input/landmark-recognition-2020/train/7/0/0/7000542ecac029aa.jpg'
a= fig.add_subplot(2,3,3)
a.set_title(index.split("/")[-1])
plt.imshow(plt.imread(index))

index= '../input/landmark-recognition-2020/train/7/0/5/7050308f31e8f117.jpg'
a= fig.add_subplot(2,3,4)
a.set_title(index.split("/")[-1])
plt.imshow(plt.imread(index))

index= '../input/landmark-recognition-2020/train/7/0/a/70a039ff5015a267.jpg'
a= fig.add_subplot(2,3,5)
a.set_title(index.split("/")[-1])
plt.imshow(plt.imread(index))

index= '../input/landmark-recognition-2020/train/7/0/f/70f0333a732c666d.jpg'
a= fig.add_subplot(2,3,6)
a.set_title(index.split("/")[-1])
plt.imshow(plt.imread(index))

plt.show()
    
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate, ElasticTransform,ChannelShuffle
)
albumentation_list =  [
    HorizontalFlip(p=0.5),
    RandomContrast(limit=0.2, p=1),
    RandomGamma(gamma_limit=(80, 120), p=1),
    RandomBrightness(limit=0.2, p=0.5),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1, 
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8), 
    ChannelShuffle(p=1),
    ElasticTransform(p=1,border_mode=cv2.BORDER_REFLECT_101,alpha_affine=40)
]
chosen_image= plt.imread('../input/landmark-recognition-2020/train/0/4/3/04305edef6cf2186.jpg')
img_matrix_list = []
bboxes_list = []
for aug_type in albumentation_list:
    img = aug_type(image = chosen_image)['image']
    img_matrix_list.append(img)
    
img_matrix_list.insert(0,chosen_image)    

titles_list = ["Original","Horizontal Flip","Random Contrast","Random Gamma","RandomBrightness",
               "Shift Scale Rotate","Channel Shuffle", "Elastic Transform"]

def plot_multiple_img(img_matrix_list, title_list, ncols, main_title="Data Augmentation"):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=2, ncols=ncols, squeeze=True)
    fig.suptitle(main_title, fontsize = 30)
    #fig.subplots_adjust(wspace=0.3)
    #fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
    plt.show()
plot_multiple_img(img_matrix_list, titles_list, ncols = 4)
