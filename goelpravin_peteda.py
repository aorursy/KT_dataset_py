# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Let's understand some of the input data first
import pandas as pd


# Path of the file to read
train_file_path = '../input/train/train.csv'

train_data = pd.read_csv(train_file_path)
# train_data.info()
corrColumns = ['Type','Age','Health','Breed1']
adoptionSpeedCorr = {}
for pet_column in train_data[corrColumns]:
    print(pet_column, train_data['AdoptionSpeed'].corr(train_data[pet_column]))
# for (petColumn in train_data.columns()):
#     print(train_data['AdoptionSpeed'].corr(train_data[petColumn]))
# what is the correlation of adoption speed with various columns
# pairwiseCorrs = train_data.corr()
# pairwiseCorrs.plot(figsize=(14,4))
train_data.plot(figsize=(15,4))