import pandas as pd

import numpy as np



import plotly.express as px



import glob



import matplotlib.pyplot as plt

import seaborn as sns



import cv2
df_train= pd.read_csv("../input/landmark-recognition-2020/train.csv")

df_submission = pd.read_csv("../input/landmark-recognition-2020/sample_submission.csv")

TRAIN_PATH = "../input/landmark-recognition-2020/train"

TEST_PATH = "../input/landmark-recognition-2020/test"
train_list = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')

test_list = glob.glob('../input/landmark-recognition-2020/test/*/*/*/*')



print( 'Images in Train Folder:', len(train_list))

print( 'Images in Test Folder:', len(test_list))
df_train.head()
df_train.shape[0], df_train.id.nunique()
df_train.landmark_id.nunique()
df_train.isna().sum()
df_image_counts = df_train.groupby("landmark_id").agg(images = ("id","nunique")).reset_index()

df_image_counts.head()
px.box(df_image_counts, x= "images",width=1000, height=300)
def plot_images(image_list,rows,cols,title):

    fig,ax = plt.subplots(rows,cols,figsize = (25,5*rows))

    ax = ax.flatten()

    for i, image_id in enumerate(image_list):

        image = cv2.imread(TRAIN_PATH+'/{}/{}/{}/{}.jpg'.format(image_id[0],image_id[1],image_id[2],image_id))

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        ax[i].imshow(image)

        ax[i].set_axis_off()

        ax[i].set_title(image_id)

    plt.suptitle(title)
plot_images(df_train.loc[df_train.landmark_id==df_image_counts[df_image_counts.images == 6272]["landmark_id"].values[0],"id"].values[:10],2,5,"Images of Landmark - 138982 (10 out of 6k images of this landmark)")
plot_images(df_train.loc[df_train.landmark_id==df_image_counts[df_image_counts.images == 2231]["landmark_id"].values[0],"id"].values[:10],2,5,"Images of Landmark with second highest number of images (10 out of 2 k images of this landmark)")
plot_images(df_train.loc[df_train.landmark_id==df_image_counts[df_image_counts.images == 10]["landmark_id"].values[0],"id"].values[:10],2,5,"Images of Landmark:" + str(df_image_counts[df_image_counts.images == 10]["landmark_id"].values[0]))