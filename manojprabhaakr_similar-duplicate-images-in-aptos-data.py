# Credit to this kernel in Whale Identification - https://www.kaggle.com/stehai/duplicate-images. 
import pandas as pd 

import os

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from PIL import Image

import imagehash



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
train.id_code.nunique()
base_image_dir = os.path.join('..', 'input/') # Joining the base directory Input

train_dir = os.path.join(base_image_dir,'train_images/') # Training Directory Location

df = pd.read_csv(os.path.join(base_image_dir, 'train.csv')) # Reading the training file

df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x))) # Getting the path of the image

df = df.drop(columns=['id_code']) # Dropping the Id code column

df = df.sample(frac=1).reset_index(drop=True) 

df.head(10) # Getting the top 10 records
def getImageMetaData(file_path):

    with Image.open(file_path) as img:

        img_hash = imagehash.phash(img)

        return img.size, img.mode, img_hash



def get_train_input():

    train_input = df.copy()

        

    m = train_input.path.apply(lambda x: getImageMetaData(x))

    train_input["Hash"] = [str(i[2]) for i in m]

    train_input["Shape"] = [i[0] for i in m]

    train_input["Mode"] = [str(i[1]) for i in m]

    train_input["Length"] = train_input["Shape"].apply(lambda x: x[0]*x[1])

    train_input["Ratio"] = train_input["Shape"].apply(lambda x: x[0]/x[1])

    

    

    img_counts = train_input.path.value_counts().to_dict()

    train_input["Id_Count"] = train_input.path.apply(lambda x: img_counts[x])

    return train_input



train_input = get_train_input()

train_input1 = train_input[['Hash']] # Getting the Hash from the new data
train_input1['New']=1 # Creating a dummy column 1
train_input2 = train_input1.groupby('Hash').count().reset_index() # Grouping the column by Hash to aggregate at Hash level
train_input2 = train_input2[train_input2['New']>1] # Filtering those instances where the hash is occuring multiple times
train_input2.shape # Checking the shape
train_input2 = train_input2.sort_values('Hash') # Sorting the data by Hash 
train_input2.head(5) # Checking the top 5 records
train_input[train_input['Hash']=='808d0f3513f33f2d'] # FIltering only one Hash from master data
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from IPython import display

import time



%matplotlib inline



PATH = "../input/train_images/1632c4311fc9.png"

image = mpimg.imread(PATH) # images are color images

plt.imshow(image);
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from IPython import display

import time



%matplotlib inline



PATH = "../input/train_images/a75bab2463d4.png"

image = mpimg.imread(PATH) # images are color images

plt.imshow(image);