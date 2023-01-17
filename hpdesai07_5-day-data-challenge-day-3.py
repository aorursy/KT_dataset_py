import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind



data = pd.read_csv('../input/cereal.csv')

dataframes = data.describe()

ttest = ttest_ind(dataframes['sodium'],dataframes['sugars'],equal_var=False)

plt.plot(ttest)