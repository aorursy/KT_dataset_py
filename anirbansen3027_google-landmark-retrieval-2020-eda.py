import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import cv2



import glob
train_data = pd.read_csv("/kaggle/input/landmark-retrieval-2020/train.csv")

TRAIN_FOLDER = '/kaggle/input/landmark-retrieval-2020/train/'
train_data.head()
train_data.landmark_id.nunique(),train_data.shape[0]
no_of_images = train_data.groupby("landmark_id").agg(count_images = ("id","nunique")).reset_index()

sns.distplot(no_of_images.count_images,kde = False,bins = 60)
no_of_images.count_images.describe()
no_of_images.sort_values(by = ["count_images"]).head()
def plot_images(image_list,rows,cols,title):

    fig,ax = plt.subplots(rows,cols,figsize = (25,5*rows))

    ax = ax.flatten()

    for i, image_id in enumerate(image_list):

        image = cv2.imread(TRAIN_FOLDER+'{}/{}/{}/{}.jpg'.format(image_id[0],image_id[1],image_id[2],image_id))

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        ax[i].imshow(image)

        ax[i].set_axis_off()

        ax[i].set_title(image_id)

    plt.suptitle(title)
plot_images(train_data.loc[train_data.landmark_id==183721,"id"].values,1,2,"Images of Landmark - 183721 (Only 2 images for this landmark)")
plot_images(train_data.loc[train_data.landmark_id==103191,"id"].values[:5],1,5,"Images of Landmark - 103191 (5 out of 13 images of this landmark)")
plot_images(train_data.loc[train_data.landmark_id==138982,"id"].values[:10],2,5,"Images of Landmark - 138982 (10 out of 6k images of this landmark)")
test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')

index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')



print( 'Images in Text Folder:', len(test_list))

print( 'Images in Index Folder:', len(index_list))