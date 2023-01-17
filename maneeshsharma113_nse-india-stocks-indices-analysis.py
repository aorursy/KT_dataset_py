import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
%matplotlib inline
df1=pd.read_csv('../input/banknifty.csv')
df2=pd.read_csv('../input/nifty50.csv')
df1.head()
df2.head()
df1.info()
df2.info()
df1['date']=pd.to_datetime(df1['date'].astype(str), format='%Y%m%d')
df2['date']=pd.to_datetime(df2['date'].astype(str), format='%Y%m%d')
del df1['index']
del df2['index']
date_df1=df1.groupby('date').mean()
date_df2=df2.groupby('date').mean()
date_df1.head()
date_df1.describe()
date_df2.head()

plt.figure(figsize=(20,7))
plt.plot(date_df1,color='blue')
plt.title('Bank Nifty Graph')
plt.figure(figsize=(20,7))
plt.plot(date_df2,color='blue')
plt.title('Nifty50 Graph')
sns.kdeplot(data=df1['high'],gridsize=50)
sns.kdeplot(data=df2['high'],gridsize=50)
sns.heatmap(df1.corr(),annot=True)
sns.heatmap(df2.corr(),annot=True)
