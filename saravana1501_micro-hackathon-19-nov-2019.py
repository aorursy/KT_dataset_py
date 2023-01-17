# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



home_data = pd.read_csv("../input/microhackathon19nov/train.csv", low_memory=False)

print('Setup Completed')



# Any results you write to the current directory are saved as output.
# split your data here!

# Specify your feature column here.

feature_columns = []

# Uncomment below line and add your code to split data.

# train_X, val_X, train_y, val_y = 



print('Split data completed')
# specify and fit the model here!

# Uncomment below line and create your model.

# model = 

print('Model creation completed')