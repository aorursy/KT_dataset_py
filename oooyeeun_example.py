# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
base_dir = "/kaggle/input/2ndparrot/"

sub = pd.read_csv(base_dir+"submission.csv")

sub = sub.drop(['Unnamed: 0'], axis = 1)

sub.head()
list_classes = ['bear', 'cat', 'raccoon', 'dog', 'rat', 'seal', 'bird', 'deer']
data_dir = base_dir + "2020_parrot_dataset/2020_parrot_dataset/"

for i in range(8):

    temp = len(os.listdir(data_dir + 'train/' + list_classes[i]))

    print("num or class {} : {}".format(list_classes[i], temp))
from PIL import Image

import matplotlib.pyplot as plt

import random



%matplotlib inline
fig, ax= plt.subplots(nrows = 8, figsize = (5, 20))

for i in range(8):

    temp_dir = data_dir + 'train/' + list_classes[i] +'/'

    temp_img = Image.open(temp_dir + os.listdir(temp_dir)[random.randint(0, 1000)])

    ax[i].imshow(temp_img)

    ax[i].set_title(list_classes[i])

    ax[i].set_xticks([])

    ax[i].set_yticks([])