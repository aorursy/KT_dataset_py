# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from random import randint

from tqdm import tqdm

import cv2

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/labels.csv")

df_train.head(10)

ax=pd.value_counts(df_train['breed'],ascending=True).plot(kind='barh',

                                                       fontsize="40",

                                                       title="Class Distribution",

                                                       figsize=(50,100))

ax.set(xlabel="Images per class", ylabel="Classes")

ax.xaxis.label.set_size(40)

ax.yaxis.label.set_size(40)

ax.title.set_size(60)

def plot_images(images, classes):

    assert len(images) == len(classes) == 9

    

    # Create figure with 3x3 sub-plots.

    fig, axes = plt.subplots(3, 3,figsize=(60,60),sharex=True)

    fig.subplots_adjust(hspace=0.3, wspace=0.3)

   

    for i, ax in enumerate(axes.flat):

        # Plot image.

        

        ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB).reshape(img_width,img_height,3), cmap='hsv')    

        xlabel = "Breed: {0}".format(classes[i])

    

        # Show the classes as the label on the x-axis.

        ax.set_xlabel(xlabel)

        ax.xaxis.label.set_size(60)

        # Remove ticks from the plot.

        ax.set_xticks([])

        ax.set_yticks([])

    

    # Ensure the plot is shown correctly with multiple plots

    # in a single Notebook cell.

    

    plt.show()
img_width=250

img_height=250

images=[]

classes=[]

#load training images

for f, breed in tqdm(df_train.values):

    img = cv2.imread('../input/train/{}.jpg'.format(f))

    classes.append(breed)

    images.append(cv2.resize(img, (img_width, img_height)))


# select random images

random_numbers = [randint(0, len(images)) for p in range(0,9)]

images_to_show = [images[i] for i in random_numbers]

classes_to_show = [classes[i] for i in random_numbers]

print("Images to show: {0}".format(len(images_to_show)))

print("Classes to show: {0}".format(len(classes_to_show)))



#plot the images

plot_images(images_to_show, classes_to_show)

   