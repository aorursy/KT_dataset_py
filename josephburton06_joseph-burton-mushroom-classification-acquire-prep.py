import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('mushrooms.csv')
df.shape
df.head().T
df.rename(index=str, columns={'class':'e_or_p'}, inplace=True)
df.isnull().sum()
train, test = train_test_split(df, test_size=.3, random_state=123, stratify=df[['e_or_p']])
train.shape
test.shape
train.apply(lambda x: x.value_counts()).T.stack()
train.groupby('odor')['e_or_p'].value_counts()
train[train.odor == 'n'].groupby('cap-shape')['e_or_p'].value_counts()
train[train.odor == 'n'].groupby('habitat')['e_or_p'].value_counts()
train[train.odor == 'n'].groupby('gill-color')['e_or_p'].value_counts()
train[train.odor == 'n'].groupby(['gill-color', 'habitat'])['e_or_p'].value_counts()