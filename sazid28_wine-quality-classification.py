%matplotlib inline
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sklearn
data = pd.read_csv("../input/winequality-red.csv")
data.head(5)
data.describe
data.columns
data.hist
data.corr

