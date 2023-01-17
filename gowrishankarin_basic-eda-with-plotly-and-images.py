# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(len(os.listdir("../input/test")))



DEV = False

# Any results you write to the current directory are saved as output.
truncated_df = pd.read_csv("../input/sample_truncated_submission.csv")

empty_df = pd.read_csv("../input/sample_empty_submission.csv")

truncated_df.head()
import matplotlib.pyplot as plt

from PIL import Image



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 9999

pd.options.display.float_format = '{:20, .2f}'.format
from PIL import Image

img = Image.open("../input/test/d390310a4ce1c08a.jpg")

img
SAMPLES_TO_EXAMINE = 15

import cv2

import time

from PIL import Image



def render_images(files):

    plt.figure(figsize=(50, 50))

    row = 1

    for an_image in files:

        image = cv2.imread(an_image)[..., [2, 1, 0]]

        plt.subplot(6, 5, row)

        plt.imshow(image)

        row += 1

    plt.show()



def read_files(files):

    images = []

    shapes = []

    for an_image in files:

        #image = Image.imread(an_image)[..., [2, 1, 0]]

        #images.append(image)

        image = Image.open(an_image)

        shapes.append((image.size[0], image.size[1], image.layers))

        

    return images, shapes



def get_images(task, files):

    start_time = time.time()

    images, shapes = read_files(files)

    end_time = time.time()

    print("Task: {0}, Duration: {1}, Image Count: {2}, Shape Count: {3}".format(task, end_time - start_time, len(images), len(shapes)))



    df = pd.DataFrame({

        'ImageID': [file.split('/')[3].split('.')[0] for file in files],

        'ImageHeight': [a_shape[0] for a_shape in shapes],

        'ImageWidth': [a_shape[1] for a_shape in shapes],

        'channels': [a_shape[2] for a_shape in shapes]

    })

    

    df["location"] = "../input/test/" + df["ImageID"] + ".jpg"

    

    return images, df

#files_to_examine = test_image_files.sample(SAMPLES_TO_EXAMINE)

#render_images(files_to_examine.files.values)
#import jpeg4py as jpeg

import glob

import time

import random

start_time = time.time()

sorted_files = sorted(glob.glob("../input/test/" + "*.jpg"))

print(len(sorted_files))

if(DEV == False): 

    small_files = sorted_files[:30000]

    medium_files = sorted_files[30000:60000]

    large_files = sorted_files[60000:90000]

    very_large_files = sorted_files[90000:]

else:

    small_files = sorted_files[:1000]

    medium_files = sorted_files[1000:2000]

    large_files = sorted_files[2000:3000]

    very_large_files = sorted_files[3000:4000]



len(very_large_files)
small_images, small_df = get_images("Read Small Files", small_files)

small_df["size"] = "small"

del small_images
medium_images, medium_df = get_images("Read Medium Files", medium_files)

medium_df["size"] = "medium"

del medium_images
large_images, large_df = get_images("Read Large Files", large_files)

large_df["size"] = "large"

del large_images
very_large_images, very_large_df = get_images("Read Very Large Files", very_large_files)

very_large_df["size"] = "vlarge"

del very_large_images
len(small_files) + len(medium_files) + len(large_files) + len(very_large_files)

images_df = pd.concat([small_df, medium_df, large_df, very_large_df])

images_df.head()
height_distribution = pd.DataFrame(images_df.ImageHeight.value_counts())

height_distribution.reset_index(inplace=True)



width_distribution = pd.DataFrame(images_df.ImageWidth.value_counts())

width_distribution.reset_index(inplace=True)

height_distribution.head()
import plotly_express as px

px.histogram(height_distribution, x="index", y="ImageHeight", 

            height=600, width=800)
px.histogram(width_distribution, x="index", y="ImageWidth", 

            height=600, width=800)
images_df['ratio'] = np.round(images_df['ImageWidth'].divide(images_df['ImageHeight'], fill_value=1))

px.scatter(images_df, x="ImageWidth", y="ImageHeight", 

           color="ratio", height=1000, width=800, 

           marginal_x="histogram", marginal_y="histogram")
print(len(images_df))

files_to_examine = random.sample(medium_files, SAMPLES_TO_EXAMINE)

render_images(files_to_examine)