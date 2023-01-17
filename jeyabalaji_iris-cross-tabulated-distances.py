import pandas as pd

import numpy as np
data = pd.read_csv('../input/iris.csv')

data = np.array(data.iloc[:,:4])

print(data.shape)
cross_tab_data = np.apply_along_axis(lambda a, b : b - a, 1, data, data)

cross_tab_data = cross_tab_data.reshape(-1,4)
cross_tab_data.shape
dists = np.apply_along_axis(np.linalg.norm, 1, cross_tab_data)
mean_dists = np.mean(dists)

print("The mean of cross-tabulated Euclidean distances in the Iris dataset is {}.".format(mean_dists))