import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
df_cov = pd.read_csv('../input/dsc-summer-school-data-visualization-challenge/2019_nCoV_20200121_20200206.csv', infer_datetime_format=True, parse_dates=['Last Update'])

df_cov.head()
df_cov.describe()
df1 = df_cov[~df_cov['Province/State'].isnull()]

df2 = df1[~df1['Confirmed'].isnull()]

df3 = df2[~(df2['Country/Region'] != 'Mainland China')]

df3

df3.nunique()
sns.set_style('whitegrid')

sns.set_palette(sns.color_palette('Greys'))

sns.barplot(x='Confirmed', y='Province/State', data=df3.head(5));



plt.title('Top 5 Confirmed COVID-19 in Mainland China')

plt.rcParams["figure.figsize"] = (15,9)