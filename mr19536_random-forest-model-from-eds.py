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
from sklearn.model_selection import train_test_split # 
from sklearn import preprocessing # for data manipulation 

from sklearn.ensemble import RandomForestRegressor # The Model

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import  GridSearchCV # For Cross-validatin 

from sklearn.metrics import mean_squared_error, r2_score #For Perfoemance Evaluation

from sklearn.externals import joblib # for model storage 

dataset_url = 'kaggle datasets download -d mr19536/wine-snob-tutorial'
data = pd.read_csv(dataset_url)
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url)

