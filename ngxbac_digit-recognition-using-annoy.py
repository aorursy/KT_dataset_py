# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from annoy import AnnoyIndex
from tqdm import tqdm

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Y_train = train["label"]
# Remove the labels, keep pixel values only
X_train = train.drop(labels = ["label"], axis = 1).values
feature_size = 28 * 28
annoy_train = AnnoyIndex(feature_size)  # Length of item vector that will be indexed
for i, x_train in enumerate(X_train):
    annoy_train.add_item(i, x_train)
# There is a trade off between speed and accuracy here
number_of_tree = 1000 # number_of_tree = 10 will get 95.5 LB
annoy_train.build(number_of_tree)
# Save to disk for fast access and optimize memory
annoy_train.save('train.ann')
# Create for test
annoy_test = AnnoyIndex(feature_size)
annoy_test.load('train.ann')
test = test.values
Y_train = Y_train.values
predict = []
for t in tqdm(test):
    nnb_idx = annoy_test.get_nns_by_vector(t, 1)
    label = Y_train[nnb_idx]
    predict.append(label[0])
predict = pd.Series(predict,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)
submission.to_csv("mnist_annoy.csv",index=False)
submission.head()
