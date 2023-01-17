import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import pandas_profiling as pp

import seaborn as sns

import warnings

import os



warnings.filterwarnings('ignore')

%matplotlib inline





print(os.listdir("../input"))
df_iris = pd.read_csv('../input/Iris.csv')
df_iris.info()
df_iris.describe()
pp.ProfileReport(df_iris)