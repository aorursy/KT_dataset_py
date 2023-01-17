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
train = pd.read_csv("../input/train.csv")
print(train.head())
labels = train.label.unique()
for label in labels:
    print(label)
req_dataset  = train[train["label"].isin(["Fireworks","Gunshot_or_gunfire", "Shatter" ])]
req_dataset.head(10)
manually_verified_labels =  req_dataset[req_dataset.manually_verified == 1]
manually_verified_labels.shape
not_verified_labels =  req_dataset[req_dataset.manually_verified == 0]
not_verified_labels.shape
not_verified_labels.to_csv("notverified.csv")
