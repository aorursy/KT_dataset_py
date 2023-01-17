# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from PIL import Image
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
im_sample = Image.open('../input/amer_sign3.png')
im_sample
df_train = pd.read_csv('../input/sign_mnist_train.csv')
df_test = pd.read_csv('../input/sign_mnist_test.csv')

print("df_train.shape:", df_train.shape)
print("df_test.shape:", df_test.shape)
print(df_train.columns)
x_train = df_train.iloc[:, 1:785].values.reshape(27455, 28, 28).astype(np.uint8)
y_train = df_train.iloc[0].values.astype(np.uint8)
x_test = df_test.iloc[:, 1:785].values.reshape(7172, 28, 28).astype(np.uint8)
y_test = df_test.iloc[0].values.astype(np.uint8)
print('x_test.shape:', x_test.shape)
print('y_test.shape:', y_test.shape)
print('x_test.dtype:', x_test.dtype)
print('y_test.dtype:', y_test.dtype)