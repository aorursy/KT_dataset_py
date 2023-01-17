import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('athletes.csv')
df.head()
df.shape
df.isnull().sum()
df.dropna(inplace=True)
df.shape
df.dtypes
df['medal_or_nm'] =  df['gold'] + df['silver'] + df['bronze']
df_medals = df[df.medal_or_nm >= 1]

df_medals.shape
df_medals.head()
df_medals.groupby('nationality')['medal_or_nm'].count()
country_count = pd.DataFrame(df_medals.groupby('nationality')['medal_or_nm'].agg('sum'))

country_count.columns = ['country_count']
df_medals = df_medals.merge(country_count, on='nationality')
df_medals.head(10)
df_medals = df_medals[df_medals.country_count > 50]
df_medals.shape
df_medals.nationality.nunique()
train, test = train_test_split(df_medals, test_size=.3, random_state=123, stratify=df_medals[['nationality']])