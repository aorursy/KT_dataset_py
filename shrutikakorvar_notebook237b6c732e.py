import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/data-visualization/country_wise_latest.csv')
df.head()
sns.countplot(x='Confirmed',data=df)
sns.lineplot(x='Active',y='Deaths', data=df)
df.plot.scatter('Active','Confirmed')
df['Active'].hist(bins=8)