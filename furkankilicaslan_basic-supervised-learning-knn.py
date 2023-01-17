# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
meyveler = pd.read_table('/kaggle/input/fruits-with-colors-dataset/fruit_data_with_colors.txt')

meyveler
meyveler.info()
meyve_degiskenler = ['mass', 'width', 'height', 'color_score']

X = meyveler[meyve_degiskenler]

y = meyveler['fruit_label']

meyve_hedefler = ['apple', 'mandarin', 'orange', 'lemon']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
knn_model = KNeighborsClassifier()

knn_model.fit(X_train_scaled, y_train)

knn_model.score(X_train_scaled, y_train)