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
df_cov['Province/State'].unique()
sns.barplot(x='Province/State', y='Death', data=df_cov)

plt.title('Average Death of Each Province/State')
plt.rcParams["figure.figsize"] = (100,100)
plt.xticks(rotation=90)
sns.barplot(x='Province/State', y='Suspected', data=df_cov)

plt.title('Average Suspected of Each Province/State')
plt.xticks(rotation=90)
sns.barplot(x='Province/State', y='Recovered', data=df_cov)

plt.title('Average Recovered of Each Province/State')
plt.xticks(rotation=90)
