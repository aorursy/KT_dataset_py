import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv(r'../input/videogamesales/vgsales.csv')
df.head()
df.shape
df.columns
df.info()
df.describe()
df.describe(include='object')
df.nunique()
df['Platform'].unique()
df['Platform'].value_counts()
df['Platform'].value_counts(normalize='True')