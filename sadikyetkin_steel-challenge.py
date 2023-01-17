import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

df.head(10)
df.info()

df.describe().T
df.isnull().values.any()