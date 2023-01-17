# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import seaborn as sns

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



feature_cols = [col for col in train.columns if col not in ['ID_code', 'target']]
for l in range(1,9):

    

    fig, ax = plt.subplots(5,5,figsize=(16,16))



    i = 0

    cols_to_plot = feature_cols[25*l-25:25*l]

        

    for row in ax:

        for col in row:

            sns.distplot(train[cols_to_plot[i]].value_counts(), ax=col, axlabel=f'{cols_to_plot[i]} binned')

            i += 1

    plt.show()
for l in range(1,9):

    

    fig, ax = plt.subplots(5,5,figsize=(16,16))

    

    i = 0        

    cols_to_plot = feature_cols[25*l-25:25*l]



    for row in ax:

        for col in row:

            sns.distplot(test[cols_to_plot[i]].value_counts(), ax=col, axlabel=f'{cols_to_plot[i]} binned')

        

            i += 1

    plt.show()