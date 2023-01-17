# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import matplotlib.patches as patches

from PIL import Image

from ast import literal_eval



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
PATH = "/kaggle/input/global-wheat-detection/"

train_csv = pd.read_csv(PATH + "train.csv")
train_csv.head()
for i in range(0, 10):

    location = i*30



    img_path = PATH + "train/" + train_csv["image_id"].iloc[location] + ".jpg"

    bbox = train_csv["bbox"].iloc[location]

    print(img_path)

    

    bbox = literal_eval(bbox)



    im = np.array(Image.open(img_path), dtype=np.uint8)

    im = im[:, :, 0]

    im[im<100] = 0

    print(im)

    

    # Create figure and axes

    fig,ax = plt.subplots(1)



    # Display the image

    ax.imshow(im, cmap = 'hot')



    # Create a Rectangle patch

    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='y',facecolor='none')



    # Add the patch to the Axes

    ax.add_patch(rect)



    plt.show()