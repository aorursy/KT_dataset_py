%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set seaborn style
sns.set_style("ticks")

# read in data
data_raw = pd.read_csv("../input/loan.csv")
data_raw.head(n = 25)
data_raw.shape

# initial plots
sns.distplot(data_raw['loan_amnt']);
sns.distplot(data_raw['funded_amnt'])
sns.jointplot(x = 'loan_amnt', y = 'funded_amnt', data = data_raw)