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
import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import glob

import os



%matplotlib inline


train_df = pd.read_csv('../input/landmark-retrieval-2020/train.csv')

train_df.shape,train_df.info()
train_df.head(10)
train_df.hist(column='landmark_id')
Values_Count=train_df['landmark_id'].value_counts()
Values_Count
test_list = glob.glob('../input/landmark-retrieval-2020/test/*/*/*/*')

index_list = glob.glob('../input/landmark-retrieval-2020/index/*/*/*/*')
print( 'Query', len(test_list), ' test images in ', len(index_list), 'index images')
plt.rcParams["axes.grid"] = False

f, axarr = plt.subplots(4, 3, figsize=(28, 24))



curr_row = 0

for i in range(12):

    example = cv2.imread(test_list[i])

    example = example[:,:,::-1]

    

    col = i%4

    axarr[col, curr_row].imshow(example)

    if col == 3:

        curr_row += 1

            

#     plt.imshow(example)

#     plt.show()