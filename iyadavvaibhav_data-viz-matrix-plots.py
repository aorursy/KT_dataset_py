import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
flights = pd.read_csv('../input/flights.csv')
flights.head()
tips = pd.read_csv('../input/tips.csv')
tips.head()
# Heatmaps

tips.corr()
sns.heatmap(tips.corr())
sns.heatmap(tips.corr(),cmap='coolwarm',annot=True)
pvflights = flights.pivot_table(values='passengers',index='month',columns='year')
pvflights
sns.heatmap(pvflights)
sns.heatmap(pvflights,cmap='magma',linecolor='white',linewidths=1)
sns.clustermap(pvflights)
sns.clustermap(pvflights,cmap='coolwarm',standard_scale=1)
