import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df2 = pd.read_csv('../input/vertebralcolumndataset/column_3C.csv')

df2.describe()

sns.pairplot(df2, hue="class", size=4, diag_kind="kde")