# Much of this work is inspired by the wonderful kernel 'Train Simple XRay CNN' by Kevin Mader

# Some code fragments are sourced or adapted directly from this Kernel

# I cite his work when appropriate, including URL/Dates, and it can also be referenced here: https://www.kaggle.com/kmader/train-simple-xray-cnn



# Much of my thinking is also guided by Google's nice explanation of AutoML for Vision. General principles are quite useful

# https://cloud.google.com/vision/automl/docs/beginners-guide



# Lastly, I found this article on how AI is changing radiology imaging quite interesting

# https://healthitanalytics.com/news/how-artificial-intelligence-is-changing-radiology-pathology
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# load help packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # additional plotting functionality



# Input data files are available in the "../input/" directory.

# For example, running the below code (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))
# load data

xray_data = pd.read_csv('../input/Data_Entry_2017.csv')



# see how many observations there are

num_obs = len(xray_data)

print('Number of observations:',num_obs)



# examine the raw data before performing pre-processing

xray_data.head(5) # view first 5 rows

#xray_data.sample(5) # view 5 randomly sampled rows
# had to learn this part from scratch, hadn't gone so deep into file paths before!

# looked at glob & os documentation, along with Kevin's methodology to get this part working

# note: DON'T adjust this code, it's short but took a long time to get right

# https://docs.python.org/3/library/glob.html

# https://docs.python.org/3/library/os.html

# https://www.geeksforgeeks.org/os-path-module-python/ 

    

from glob import glob

#import os # already imported earlier



my_glob = glob('../input/images*/images/*.png')

print('Number of Observations: ', len(my_glob)) # check to make sure I've captured every pathway, should equal 112,120
# Map the image paths onto xray_data

# Credit: small helper code fragment adapted from Kevin Mader - Simple XRay CNN on 12/09/18

# https://www.kaggle.com/kmader/train-simple-xray-cnn

full_img_paths = {os.path.basename(x): x for x in my_glob}

xray_data['full_path'] = xray_data['Image Index'].map(full_img_paths.get)
# Explore the dataset a bit



# Q: how many unique labels are there? A: many (836) because of co-occurence

# Note: co-occurence will turn out to be a real pain to deal with later, but there are several techniques that help us work with it successfully

num_unique_labels = xray_data['Finding Labels'].nunique()

print('Number of unique labels:',num_unique_labels)



# let's look at the label distribution to better plan our next step

count_per_unique_label = xray_data['Finding Labels'].value_counts() # get frequency counts per label

df_count_per_unique_label = count_per_unique_label.to_frame() # convert series to dataframe for plotting purposes



print(df_count_per_unique_label) # view tabular results

sns.barplot(x = df_count_per_unique_label.index[:20], y="Finding Labels", data=df_count_per_unique_label[:20], color = "green"), plt.xticks(rotation = 90) # visualize results graphically
# define dummy labels for one hot encoding - simplifying to 14 primary classes (excl. No Finding)

dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 

'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'] # taken from paper



# One Hot Encoding of Finding Labels to dummy_labels

for label in dummy_labels:

    xray_data[label] = xray_data['Finding Labels'].map(lambda result: 1.0 if label in result else 0)

xray_data.head(20) # check the data, looking good!
xray_data[dummy_labels]
# now, let's see how many cases present for each of of our 14 clean classes (which excl. 'No Finding')

clean_labels = xray_data[dummy_labels].sum().sort_values(ascending= False) # get sorted value_count for clean labels

print(clean_labels) # view tabular results



# plot cases using seaborn barchart

clean_labels_df = clean_labels.to_frame() # convert to dataframe for plotting purposes

sns.barplot(x = clean_labels_df.index[::], y= 0, data = clean_labels_df[::], color = "green"), plt.xticks(rotation = 90) # visualize results graphically
import pandas as pd

import numpy as np

from tqdm import tqdm_notebook as tqdm

tqdm().pandas()

import re

import torch

import string
length = 512

width = 512

channels = 3
to_be_removed = xray_data[(xray_data['Finding Labels'] == 'No Finding')].index

xray_data.drop(to_be_removed , inplace=True)


import tensorflow as tf



from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

import imgaug.augmenters as iaa

import matplotlib.pyplot as plt



import seaborn as sns



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import pickle as pkl

import h5py

import cv2

import os
train_data=xray_data.head(286)
def read_and_process_image(list_of_images):

    X = [] 

    for image in tqdm(list_of_images):

       # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         print(image)

        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (length,width), interpolation=cv2.INTER_CUBIC))  

    return X
train_images = [i for i in train_data['full_path'].tolist()]
trainX = read_and_process_image(train_images)
train_data
np.save('/kaggle/working/trainX.npy', trainX)
train_data.to_csv("/kaggle/working/chest_extra.csv")
# The AUC curve looks good - it shows much tighter results than before in terms of the spread

# Clearly demonstrates the model works and has significant predictive power!

# Great ending, stopping here