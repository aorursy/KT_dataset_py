import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
df_cov = pd.read_csv('../input/dsc-summer-school-data-visualization-challenge/2019_nCoV_20200121_20200206.csv', infer_datetime_format=True, parse_dates=['Last Update'])
df_cov.head(10)

df_cov[df_cov['Country/Region'] == 'Ma'].sort_values(by='Last Update')
df_cov.describe()
df_cov.columns.tolist()
df_cov['Country/Region'].unique()
sns.set_palette(sns.color_palette('colorblind'))
sns.despine(bottom=True, left=True)
top_country = pd.DataFrame(df_cov['Country/Region'].value_counts()[:5]).reset_index()
sns.barplot(x='Country/Region', y='index', data=top_country)
plt.ylabel('Country')
plt.xlabel('Number of COVID-19 Cases')
ch_cov = df_cov[df_cov['Country/Region'] == 'Mainland China']
ch_cov = ch_cov[ch_cov['Province/State'] == 'Shanghai'].sort_values(by='Last Update')
col = ['Confirmed', 'Recovered', 'Death', 'Last Update']
#ch_cov.head(20)
ch_cov_mi = ch_cov[col].dropna()
plt.figure(figsize=(12,6))
sns.lineplot(data=ch_cov_mi, x="Last Update", y="Confirmed", label='confirmed')
sns.lineplot(data=ch_cov_mi, x="Last Update", y="Recovered", label='recovered')
sns.lineplot(data=ch_cov_mi, x="Last Update", y="Death", label='death')
plt.legend()
plt.title('COVID-19 Situations in Shanghai')
