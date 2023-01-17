import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/avocado.csv')
df.info()
df.describe()
df.head()
df.tail(15)
df_i3 = df.iloc[3]
#df_i3 = df_i3.to_frame()

df_i18240 = df.iloc[18240]
#df_i18240 = df_i18240.to_frame()

pd.concat([df_i3,df_i18240], axis = 1, ignore_index = True)
df.corr()
f,axis = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.4, fmt= '.2f',ax=axis)
plt.show()
df.plot(kind = 'scatter', x = 'Small Bags', y = 'Total Volume', color = 'magenta', alpha = '0.6')
plt.xlabel('Small Bags')
plt.ylabel('Total Volume')
plt.title("Comparison Between Total Volume and Small Bags")
plt.show()
df.plot(kind = 'scatter', x = 'Large Bags', y = 'Total Volume', color = 'red', alpha = '0.6')
plt.xlabel('Large Bags')
plt.ylabel('Total Volume')
plt.title("Comparison Between Total Volume and Large Bags")
plt.show()
df.plot(kind = 'scatter', x = 'XLarge Bags', y = 'Total Volume', color = 'blue', alpha = '0.6')
plt.xlabel('XLarge Bags')
plt.ylabel('Total Volume')
plt.title("Comparison Between Total Volume and XLarge Bags")
plt.show()
df.region.unique()
sf = df['region'] == 'SanFrancisco'
df[sf].head()
df[sf].count()
df_exp = df[sf & (df['AveragePrice'] > 2.0)]

df_exp.count()
df_exp[df['type']=='organic'].count()
df.boxplot(column = 'AveragePrice', by = 'type', figsize = (10,10))
plt.show()
df.describe()
outliers = (df['AveragePrice'] > 2.5) | (df['AveragePrice'] < 0.26)
df[outliers].count()
df_ts = df.set_index('Date')
df_ts.head()
df_ts.tail()
df_ts.sort_index()
df_ts.tail()
#df_ts.resample("A").mean()
df.head(15)