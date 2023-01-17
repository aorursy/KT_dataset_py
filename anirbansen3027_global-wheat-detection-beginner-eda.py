# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt



import cv2



import os
BASE_DIR = '/kaggle/input/global-wheat-detection/'

train_data = pd.read_csv(BASE_DIR+"train.csv")

submission_file = pd.read_csv(BASE_DIR+"sample_submission.csv")

train_images_dir = BASE_DIR + "train/"
train_data.head()
submission_file.head()
print("The training data has {} rows and {} columns".format(train_data.shape[0],train_data.shape[1]))
train_data.isna().sum()
train_data.nunique()
train_data.source.unique()
train_data.groupby("source")["image_id"].nunique()
all_images = set(x.split(".")[0] for x in os.listdir(train_images_dir))

images_with_bb = set(train_data.image_id.unique())

images_without_bb = all_images^ images_with_bb
df_images_without_bb=pd.DataFrame(images_without_bb,columns = ["image_id"])
train_data.head()
train_data[["x_start","y_start","width","height"]] = pd.DataFrame([i[1:-1].split(',') for i in train_data.bbox.to_list()],index=train_data.index)
train_data = train_data.astype({"x_start":float,"y_start":float,"width":float,"height":float})

train_data = train_data.astype({"x_start":int,"y_start":int,"width":int,"height":int})
train_data.head()
def plot_images(image_list,rows,cols,title):

    fig,ax = plt.subplots(rows,cols,figsize = (25,5))

    ax = ax.flatten()

    for i, image_id in enumerate(image_list):

        image = cv2.imread(train_images_dir+'{}.jpg'.format(image_id))

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        ax[i].imshow(image)

        ax[i].set_axis_off()

        ax[i].set_title(image_id)

    plt.suptitle(title)
plot_images(train_data[train_data.source == 'arvalis_1'].sample(5)["image_id"].values,1,5,"Images with wheat")
plot_images(train_data[train_data.source == 'arvalis_2'].sample(5)["image_id"].values,1,5,"Images with wheat")
plot_images(train_data[train_data.source == 'arvalis_3'].sample(5)["image_id"].values,1,5,"Images with wheat")
plot_images(train_data[train_data.source == 'ethz_1'].sample(5)["image_id"].values,1,5,"Images with wheat")
plot_images(train_data[train_data.source == 'inrae_1'].sample(5)["image_id"].values,1,5,"Images with wheat")
plot_images(train_data[train_data.source == 'rres_1'].sample(5)["image_id"].values,1,5,"Images with wheat")
plot_images(train_data[train_data.source == 'usask_1'].sample(5)["image_id"].values,1,5,"Images with wheat")
plot_images(df_images_without_bb.sample(10)["image_id"].values,2,5,"Images without wheat")
def plot_images_with_bb(imageId):

    plt.rcParams["figure.figsize"] = (10,10)

    bboxes = train_data[train_data.image_id == imageId]

    image = cv2.imread(train_images_dir+'{}.jpg'.format(imageId))

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    for row in bboxes.iterrows():

        image = cv2.rectangle(image,(row[1]["x_start"],row[1]["y_start"]),(row[1]["x_start"]+row[1]["width"],row[1]["y_start"]+row[1]["height"]),(255,0,0),5)

        fig = plt.imshow(image)

    plt.axis("off")

    plt.title(imageId)
plot_images_with_bb("2ae9c276f")