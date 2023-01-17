# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv("../input/wine-customer-segmentation/Wine.csv")

dataset
dataset.info()
X = dataset.iloc[:,0:13]

y = dataset.iloc[:, 13]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)



X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

X_train
X_test
new_dataset_train = pd.DataFrame(data=X_train, columns=['PC1', 'PC2'])

new_dataset_test = pd.DataFrame(data=X_test, columns=['PC1', 'PC2'])

# Con-catenating test and train datasets

new_dataset = pd.concat([new_dataset_train.reset_index (drop=True), new_dataset_test], axis=1)

new_dataset.shape
dataset.shape