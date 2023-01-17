import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Definitions
max_factor = 1000
column = "v50"

# Load data (only some samples and no NaN)
data = pd.read_csv("../input/train.csv", usecols=[column], nrows=2000).dropna()
X = data[column].values
del data

# Function that compute the number of different values, giving a factor
def diff_values(X, factor):
    return np.unique(np.round(X*factor)).size
    
y = [diff_values(X, factor) for factor in range(max_factor)]
%matplotlib inline
plt.plot(y)
