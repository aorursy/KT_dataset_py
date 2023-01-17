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

plt.figure(figsize=(10,5))
sns.set_style('white')
chart = sns.countplot(x="Province/State", data=df_cov[df_cov['Last Update'] >= '20200101'])

chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=90, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large')
