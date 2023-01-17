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
#Import semua library yang dibutuhkan 

import pandas as pd # Untuk memanipulasi file csv

import numpy as np # Untuk melakukan proses aritmetika pada matriks

import matplotlib.pyplot as plt # Untuk visualisasi pada dataset



# Membelah dataset menjadi train dan test, yang kedua untuk mencari hyperparameter yang baik

from sklearn.model_selection import train_test_split, GridSearchCV

# Model KNN nya sendiri

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

# Melakukan proses standardisasi pada suatu dataset 

from sklearn.preprocessing import StandardScaler

# Memberi label untuk tipe data kategorik

from sklearn.preprocessing import LabelEncoder

# Membantu dalam hal mengisi missing value

from sklearn.preprocessing import Imputer

# Mengukur akurasi model terhadap data yang diketahui

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# Import dataset ke Python

df = pd.read_csv('../input/heart.csv')

df.head() # Melihat beberapa baris pertama dari dataset