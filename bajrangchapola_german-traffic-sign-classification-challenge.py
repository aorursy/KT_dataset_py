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
train_data = glob.glob('../input/gtsrb_challenge/GTSRB_Challenge/train/*/*.ppm')

test_data = glob.glob('../input/gtsrb_challenge/GTSRB_Challenge/test/*.ppm')

print("Total number of training images: ", len(train_data))

print("Total number of test images: ", len(test_data))

train_data = pd.Series(train_data)

test_data = pd.Series(test_data)

# train_df: a dataframe with 2 field: Filename, ClassId

train_df = pd.DataFrame()

# generate Filename field

train_df['Filename'] = train_data.map(lambda img_name: img_name.split("/")[-1])

# generate ClassId field

train_df['ClassId'] = train_data.map(lambda img_name: int(img_name.split("/")[-2]))

train_df.head()
# train_df: a dataframe with 2 field: Filename, ClassId

test_df = pd.DataFrame()



# generate Filename field

test_df['Filename'] = test_data.map(lambda img_name: img_name.split("/")[-1])







# print(test_df)



test_df.head()
X_train, y_train = train_df['Filename'], train_df['ClassId']

X_test = test_df['Filename'] 
# Number of training examples

n_train = len(X_train)

print(n_train)



#  Number of testing examples.

n_test = len(X_test)

print(n_test)