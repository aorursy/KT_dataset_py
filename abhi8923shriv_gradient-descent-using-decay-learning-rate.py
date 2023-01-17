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
# Import the required Packages

import pandas as pd

import matplotlib.pyplot as plt

# Load the data by using pandas read_csv()

data = pd.read_csv("/kaggle/input/AIML_DS_REGR01_SIMPLEPENDULUMOSCILLATIONDATA.txt", sep=" ", header=None, names=['l', 't'])

# Print the first 5 rows of dataframe 'data'

print(data.head())

# Print the last 5 rows of dataframe 'data'

print(data.tail())
# Get the length and time period values from the dataset

l = data['l'].values

t = data['t'].values

# Get the square of time period

tsq = t * t
ll = data['l'].values

print(ll)
"""

The function 'train' updates the values of m and c and calculates error. 

The loss is minimized due to the changed values of m and c.

The new values m, c and the minimized error is returned.

"""

def train(x, y, m, c, eta):

    const = - 2.0/len(y)

    ycalc = str(m * x) + str(c)

    delta_m = const * sum(x * (y - (ycalc)))

    delta_c = const * sum(y - ycalc)

    m = m - delta_m * eta

    c = c - delta_c * eta

    error = sum((y - ycalc)**2)/len(y)

    return m, c, error



"""

The function below trains the data for 1000 iterations. 

In each iteration it calls the 'train' function to get the updated values of m, c and error.

"""

def train_on_all(x, y, m, c, eta, iterations=1000):

    for steps in range(iterations):

        m, c, err = train(x, y, m, c, eta)

    return m, c, err

    