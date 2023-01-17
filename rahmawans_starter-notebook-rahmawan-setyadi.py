import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
df_cov = pd.read_csv('../input/dsc-summer-school-data-visualization-challenge/2019_nCoV_20200121_20200206.csv', infer_datetime_format=True, parse_dates=['Last Update'])
df_cov.head()
df_cov
df_cov[~df_cov['Confirmed'].isnull()]
df_cov.describe()
sns.set_style('white')
sns.barplot(x='Country/Region', y='Confirmed', data=df_cov)
plt.xticks(rotation=90)
plt.title('Confirmed Covid in The World')
sns.barplot(x='Country/Region', y='Death', data=df_cov)
plt.xticks(rotation=90)
plt.title('Death by Covid in The World')
sns.barplot(x='Country/Region', y='Suspected', data=df_cov)
plt.xticks(rotation=90)
plt.title('Suspected Covid in The World')
sns.barplot(x='Country/Region', y='Recovered', data=df_cov)
plt.xticks(rotation=90)
plt.title('Recovered Covid in The World')