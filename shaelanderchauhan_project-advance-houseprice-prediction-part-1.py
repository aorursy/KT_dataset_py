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
# Refrence Material

import numpy as np # Linear Algebera

import pandas as pd # data preprocessiog



import matplotlib.gridspec as gridspec

from datetime import datetime

from scipy.stats import skew # A data is called as skewed when curve appears distorted or skewed either to the left or 

#to the right, in a statistical distribution



from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV ,LassoCV ,RidgeCV # Cv (cross Valadation)

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler



from sklearn.model_selection import KFold ,cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import matplotlib.style as style

import seaborn as sns 

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import missingno as msno



 



train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train.head()
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test.head()
print(f'Train Data has {train.shape[0]} Rows and {train.shape[1]} Columns')

print(f'Test  Data has {test.shape[0]} Rows and {test.shape[1]} Columns')