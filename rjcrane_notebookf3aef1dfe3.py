# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib #visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train_df.head()
#create pandas dataframes from csv files

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

#all_df = pd.concat([train_df,test_df]) #for viz purposes, I'll use all available data







train_df["LotFrontage"].hist()
