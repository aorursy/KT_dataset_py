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

df_cov.head()


sns.set(rc={'figure.figsize':(15,8)})

xAxis = (df_cov['Province/State']).head(10)





sns.barplot(x=xAxis, y='Confirmed', data=df_cov)



plt.title('Total Cases By Provinces/State')

plt.xlabel('Provinces or State')

plt.ylabel('Number of Cases')


