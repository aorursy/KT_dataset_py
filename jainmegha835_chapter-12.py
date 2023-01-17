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
from sklearn import datasets

from sklearn.feature_selection import VarianceThreshold
X=[[0,2,0,3],[0,1,4,3],[0,1,1,3]]

selector=VarianceThreshold()

selector.fit_transform(X)



from sklearn.feature_selection import VarianceThreshold

X = [[0, 1, 0],

     [0, 1, 1],

     [0, 1, 0],

     [0, 1, 1],

     [1, 0, 0]]

threshold=VarianceThreshold(threshold=(.75*(1-.75)))

threshold.fit_transform(X)
import pandas as pd

import numpy as np

features=np.array([[1,1,1],[2,2,0],[3,3,1],[4,4,0],[5,5,1]])

dataframe=pd.DataFrame(features)

corr_matrix=dataframe.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

dataframe.drop(dataframe.columns[to_drop], axis=1).head(3)

dataframe.corr()
upper
# You have a categorical target vector 

# and want to remove uninformative features
from sklearn.datasets import load_iris

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2,f_classif

iris=load_iris()

features=iris.data

target=iris.target



features=features.astype(int)

ch2_selector=SelectKBest(chi2,k=2)

features_kBest=ch2_selector.fit_transform(features,target)

print("Original number of features:", features.shape[1]) 

print("Reduced number of features:", features_kBest.shape[1])

import warnings 

from sklearn.datasets import make_regression 

from sklearn.feature_selection import RFECV 

from sklearn import datasets, linear_model

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

features, target = make_regression(n_samples = 10000,  n_features = 100, n_informative = 2,random_state=1)

ols = linear_model.LinearRegression()



rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error") 

rfecv.fit(features, target) 

rfecv.transform(features)
