# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_iris

iris_dataset = load_iris()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print('Keys: \n{}'.format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target: {}".format(iris_dataset['target']))
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: {}".format(iris_dataset['feature_names']))
print("First five rows of data: \n{}".format(iris_dataset['data'][:5]))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
# ceate dataframe from the data in X_train

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# create a satter matrix from the dataframe, colour by y_train

pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)