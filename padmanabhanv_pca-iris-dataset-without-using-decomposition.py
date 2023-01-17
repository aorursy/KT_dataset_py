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

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler 

from scipy.linalg import eigh

import seaborn as sn
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"



# load dataset into Pandas DataFrame

iris_df = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")



print(iris_df.head())
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']



data = iris_df.loc[:, features].values

target = iris_df.loc[:, ['species']].values



standardized_data = StandardScaler().fit_transform(data)



print("Shape of the data after standardized : ", standardized_data.shape)
cov_matrix = np.matmul(standardized_data.T, standardized_data)

print("Shape of cov matrix : ", cov_matrix.shape)



euigen_values, euigen_vectors = eigh(cov_matrix, eigvals=(2,3))

print("Shape of vectors : ", euigen_vectors.shape)



euigen_vectors_T = euigen_vectors.T

print("Shape of vectors transpose: ", euigen_vectors_T.shape)
result_data = np.matmul(euigen_vectors_T, standardized_data.T)

print("Shape of new Coordinates: ", result_data.shape)

print("Shape of target: ", target.shape)

# Stacking the species variable to the data.

result_data = np.vstack((result_data, target.T)).T

print("Shape of new Coordinates: ", result_data.shape)
result_df = pd.DataFrame(data=result_data, columns=('first_principal', 'second_principal', 'label'))

print(result_df.head())



sn.FacetGrid(result_df, hue='label', height=7).map(plt.scatter, 'first_principal', 'second_principal').add_legend()