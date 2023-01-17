%matplotlib inline
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
import pandas as pd
loans = pd.read_csv("../input/loan.csv")

