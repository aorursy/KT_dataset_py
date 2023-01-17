import matplotlib.pyplot as plt

import seaborn as sns

import requests

import re

import pandas as pd
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/Saudi_Hotels.csv')
df.info()
df.head()
df.describe()
ax = df['City'].value_counts().plot(kind = 'barh', title = 'Number of hotels from each city')

ax.set_ylabel('City')
df.groupby('City')['Rating'].mean().plot(kind = 'barh', title = 'Avarage rating for each city')