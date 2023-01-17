import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
results= pd.read_csv("../input/2016 County Election Data.csv")

results.info()
filtered = results[['Income-per-capita', 'Percent-white', 'Percent-in-poverty','Bachelors-degree-or-higher', 'Clinton-lead']]
pd.plotting.scatter_matrix(filtered, figsize=(20,20))

plt.show()
#correlation coefficient for education level and Clinton's lead

np.corrcoef(results['Bachelors-degree-or-higher'],results['Clinton-lead'])
#correlation coefficient for percent of people in poverty and Clinton's lead

np.corrcoef(results['Percent-in-poverty'],results['Clinton-lead'])
#correlation coefficient for percent of white people and Clinton's lead

np.corrcoef(results['Percent-white'],results['Clinton-lead'])