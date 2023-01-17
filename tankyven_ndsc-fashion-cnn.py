# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import keras

from keras import layers, models, optimizers

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.utils import resample

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import shutil

%matplotlib inline
train_data = pd.read_csv('../input/fashion.csv')
sns.countplot(x=train_data['Category'],data=train_data)

plt.title('Count Across Category Types')
# Downsample category 18,25 & 26

# Separate majority and minority classes

train_data_18 = train_data[train_data.Category==18]

train_data_25 = train_data[train_data.Category==25]

train_data_26 = train_data[train_data.Category==26]

train_data_oth = train_data[(train_data.Category!=18) & (train_data.Category!=25) & (train_data.Category!=26)]
# Downsample majority class

train_data_18_downsampled = resample(train_data_18, 

                                 replace=False,    # sample without replacement

                                 n_samples=20000,     # to match minority class

                                 random_state=123) # reproducible results

train_data_25_downsampled = resample(train_data_25, 

                                 replace=False,    # sample without replacement

                                 n_samples=20000,     # to match minority class

                                 random_state=123) # reproducible results

train_data_26_downsampled = resample(train_data_26, 

                                 replace=False,    # sample without replacement

                                 n_samples=20000,     # to match minority class

                                 random_state=123) # reproducible results



# Combine minority class with downsampled majority class

train_data = pd.concat([train_data_18_downsampled, train_data_oth])

train_data = pd.concat([train_data, train_data_25_downsampled])

train_data = pd.concat([train_data, train_data_26_downsampled])



# Display new class counts

train_data.Category.value_counts()

# 1    49

# 0    49

# Name: balance, dtype: int64
sns.countplot(x=train_data['Category'],data=train_data)

plt.title('Count Across Category Types')
cwd = os.getcwd()

print ('Current directory: {}'.format(cwd))
os.listdir('/kaggle/working/Train/Fashion/')
os.listdir("../input/fashion_image_resized/fashion_image_resized/train/")
new_folder_paths = ['Train',

                    os.path.join('Train','Fashion')]

for folder_path in new_folder_paths:

    if (os.path.isdir(folder_path) is False):

        os.mkdir(folder_path)
folder_path_dict = {i:'Fashion' for i in range(17, 31, 1)}

for category in range(17,31,1):

        

    category_img_paths = train_data[train_data['Category']==category]['image_path'].values.tolist()

    folder_path = os.path.join('Train', folder_path_dict[category], str(category))



    if (os.path.isdir(folder_path) is False):

        os.mkdir(folder_path)



    for img_path in category_img_paths:

        img_name = img_path.split('/')[1]

        corrected_img_path = "../input/fashion_image_resized/fashion_image_resized/train/"

        

        # Copy images into their appropriate category folders

        try:

            shutil.copy(os.path.join('../input/fashion_image_resized/fashion_image_resized/train/', img_name), os.path.join(folder_path, img_name))

            print('{} moved successfully'.format(img_name))

        except:

            print('{} not found'.format(img_name))

            continue
# Directories for our training & test splits

base_dir = os.path.join(os.getcwd(), 'Train', 'Fashion')

train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)

test_dir = os.path.join(base_dir, 'test')

os.mkdir(test_dir)



# Directory with our training categories

n_labels = 31

for category_id in range(17,n_labels,1):

    train_category_dir = os.path.join(train_dir, str(category_id))

    if (os.path.isdir(train_category_dir) is False):

        os.mkdir(train_category_dir)



# Directory with our test categories

for category_id in range(17,n_labels,1):

    test_category_dir = os.path.join(test_dir, str(category_id))

    if (os.path.isdir(test_category_dir) is False):

        os.mkdir(test_category_dir)