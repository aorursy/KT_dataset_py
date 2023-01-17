import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
cdbsd = pd.read_csv("../input/chicago-divvy-bicycle-sharing-data/data.csv")
cdbsd.head()
cdbsd.info()
cdbsd.shape
cdbsd.dropna(inplace = True)
cdbsd['temperature'].describe()
cdbsd['new_temperature'] = cdbsd['temperature'].round(0)
cdbsd.groupby('new_temperature').count()[['trip_id']]
cdbsd.boxplot(column='new_temperature')
cdbsd['new_temperature'].hist()