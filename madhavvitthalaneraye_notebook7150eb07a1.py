import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df=pd.read_csv('../input/indian-food-101/indian_food.csv')



df.head()
df['state'].value_counts(normalize='True')
df['state'].value_counts()
df['state'].unique()
df.nunique()
df.describe(include='object')
df.describe()
df.info()
df.columns
df.shape