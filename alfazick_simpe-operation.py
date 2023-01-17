# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from scipy import sparse
# Create a 2D NumPy array with a diagonal of ones, and zeros everywhere else

eye = np.eye(4)

print("NumPy array:\n{}".format(eye))
# Convert the NumPy array to a SciPy sparse matrix in CSR format

# Only the nonzero entries are stored

sparse_matrix = sparse.csr_matrix(eye)

print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))
data = np.ones(4)

row_indices = np.arange(4)

col_indices = np.arange(4)

eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))

print("COO representation:\n{}".format(eye_coo))
%matplotlib inline

import matplotlib.pyplot as plt



# Generate a sequence of numbers from -10 to 10 with 100 steps in between

x = np.linspace(-10,10,100)

# Create a second array using sine

y = np.sin(x)

# The plot function makes a line chart of one array against another

plt.plot( x, y, marker='x')
import pandas as pd
# create a simple dataset of people

data = {'Name':['John',"Anna","Peter","Linda"],

       'Location':['New York','Paris','Berlin','London'],

       'Age':[24, 13, 53, 33]}

data_pandas = pd.DataFrame(data)

# IPython.display allows "pretty printing" of dataframes

# in the Jupiter notebook

data_pandas
# Select all rows that have an age column greater than 30

data_pandas[data_pandas.Age>30]
import sys

print("Python version: {}".format(sys.version))

import pandas as pd

print("pandas version: {}".format(pd.__version__))

import matplotlib

print("matplotlib version: {}".format(matplotlib.__version__))

import numpy as np

print("NumPy version: {}".format(np.__version__))

import scipy as sp

print("SciPy version: {}".format(sp.__version__))

import IPython

print("IPython version: {}".format(IPython.__version__))

import sklearn

print("scikit-learn version: {}".format(sklearn.__version__))