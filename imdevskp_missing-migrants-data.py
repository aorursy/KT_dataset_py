import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/missing-migrants-project/MissingMigrants-Global-2019-12-31_correct.csv')

df.head(4)
df.info()
df.describe()
df.isna().sum()