import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/millenniumofdata_v3_headlines.csv")
data.describe()
data.shape