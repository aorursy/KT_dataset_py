import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/7160_1.csv')
df[:3]
df.isnull().sum()
df.shape
df['city'].value_counts()[:20].plot(kind='barh')
df['province'].value_counts()[:20].plot(kind='barh')
df['categories'].value_counts()[:20].plot(kind='barh')
df['postalCode'].value_counts()[:20].plot(kind='barh')
#this dataset looks like a good candidate for NLP module.. To be continued