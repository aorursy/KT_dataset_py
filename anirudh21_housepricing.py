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
data=pd.read_csv("../input/train.csv")

train_data=data.copy()

train_data.drop(["Id","SalePrice"],axis=1,inplace=True)
for i in train_data.select_dtypes("object").columns:

    train_data[i]=train_data[i].astype("category").cat.codes
for i in train_data.columns:

    if(train_data[i].isnull().sum()>0.7*1460):

        train_data.drop(i,axis=1,inplace=True)
for i in train_data.columns:

    train_data[i].fillna(value=train_data[i].mean(),inplace=True)