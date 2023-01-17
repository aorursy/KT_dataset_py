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
!pip install pycaret
from pycaret.datasets import get_data
# loading inbuilt datasets from pycaret
# we will use diaetes dataset from pycaret itself
# but yes we can also use our own csv file
diabetes_df = get_data('diabetes')
# this is a classification problem
# importing classification modules
from pycaret.classification import *
clf = setup(data = diabetes_df, target='Class variable')
# to compare the model with other algorithms available 
# compare_models() function is used
compare_models()
# regression problem
boston_df = get_data('boston')
# instead of * we can also import a particular algorithm
from pycaret.regression import *
reg = setup(data = boston_df, target = 'medv')
boston_df.describe()
compare_models()
regressor_model = create_model('catboost')
regressor_model
reg_model_hypertune = tune_model('catboost', n_iter = 50, optimize = 'mae')
from pycaret.clustering import *
jewellery = get_data('jewellery')
clust_algo = setup(jewellery)
kmeans = create_model('kmeans')
kmeans
### assign labels to the dataframe
kmeans_df = assign_model(kmeans)
# just have a look on the last column how the data points are assigned to different clusters
kmeans_df.head()
# PCA is applied before clustering
plot_model(kmeans)
