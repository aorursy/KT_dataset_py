import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt #for plotting the graphs or images
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
print(train_data.head(10))
print()
print("Here, id means Image Id and landmark_id points to a specific ID of the landmark ")
train_data.describe()
print(train_data.isna().sum())
print()
!pip install basic_image_eda
from basic_image_eda import BasicImageEDA
data_dir = "../input/landmark-recognition-2020/train/0"
extensions = ['jpg']
threads = 0
dimension_plot = True
channel_hist = True
nonzero = False
hw_division_factor = 1.0

BasicImageEDA.explore(data_dir, extensions, threads, dimension_plot, channel_hist, nonzero, hw_division_factor)
train_data['landmark_id'].value_counts()
print("Types of Landmarks: {81313}")
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

from random import randrange
fig= plt.figure(figsize=(20,10))
index= '../input/landmark-recognition-2020/train/2/3/6/23603d71816b6452.jpg'
a= fig.add_subplot(2,3,1)
a.set_title(index.split("/")[-1])
plt.imshow(plt.imread(index))

index= '../input/landmark-recognition-2020/train/7/0/4/7040a5cfa43e0633.jpg'
a= fig.add_subplot(2,3,2)
a.set_title(index.split("/")[-1])
plt.imshow(plt.imread(index))

index= '../input/landmark-recognition-2020/train/4/1/0/41000aafca574dfe.jpg'
a= fig.add_subplot(2,3,3)
a.set_title(index.split("/")[-1])
plt.imshow(plt.imread(index))

index= '../input/landmark-recognition-2020/train/4/3/1/43101b9ac11ed672.jpg'
a= fig.add_subplot(2,3,4)
a.set_title(index.split("/")[-1])
plt.imshow(plt.imread(index))

index= '../input/landmark-recognition-2020/train/4/3/1/43105797059abd97.jpg'
a= fig.add_subplot(2,3,5)
a.set_title(index.split("/")[-1])
plt.imshow(plt.imread(index))

index= '../input/landmark-recognition-2020/train/4/1/0/41008546ba23b770.jpg'
a= fig.add_subplot(2,3,6)
a.set_title(index.split("/")[-1])
plt.imshow(plt.imread(index))

plt.show()
    