import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_file = '../input/CMS-Open-Data.csv'
dataset = pd.read_csv(dataset_file)
dataset.head()
print(dataset.shape)
masses = dataset.M # assigning mass values to a variable 'mass'

# bins from 2.0 to 5.0 in 0.10 intervals
hist = masses.hist(bins=[2.0, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.0, 3.10, 3.20, 3.30, 3.40, 3.50, 3.60, 3.70, 3.80, 3.90, 4.0, 4.10, 4.20, 4.30, 4.40, 4.50, 4.60, 4.70, 4.80, 4.90, 5.0])
plt.title('The histogram of the invariant masses of two muons \n')
plt.xlabel("Invariant Masses in GeV")
plt.ylabel("Frequency")
plt.show()
mean_masses = np.mean(masses)
print(mean_masses) 

variance_masses = np.var(masses)
print(variance_masses)