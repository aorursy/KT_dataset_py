import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/winemag-data_first150k.csv',index_col=0)
df.head()

description = df['description']

df = df.drop(['description'],axis=1)
statistics_points = df['points'].groupby(df['country']).mean()
mean_value = [ statistics_points[i] for i in range(len(statistics_points))]

_country = df.country.unique().tolist()

del(_country[22])
_country = sorted(_country)

f = plt.figure(figsize=(20,10))
plt.plot( _country, mean_value,  marker='o', markerfacecolor='#5e0000', markersize=12, color='#fd742d', linewidth=4)
plt.xticks( rotation=86)
plt.tick_params(axis = 'both', labelsize = 15)
plt.grid(True)
plt.title('Average points', fontsize = 20)
plt.show()
plt.figure(figsize=(20,5))
bars = (_country)
y_pos = np.arange(len(bars))
plt.bar(y_pos, mean_value, color = '#dc0f0f')
plt.xticks(y_pos, bars)
plt.xticks( rotation=86)
plt.ylim(80,95)
plt.title('Average points', fontsize = 20)
plt.tick_params(axis = 'both', labelsize = 15)
plt.show()

f = plt.figure(figsize=(20,5))
sns.countplot(x='points',data = df, palette='hls' )
plt.show()
df.isnull().sum()
f = plt.figure(figsize=(20,5))
sns.heatmap(df.isnull(),yticklabels=False,cmap= 'viridis')
most_country = ['US','England','Canada','Austria']

f = plt.figure(figsize=(10,7))
f = sns.boxplot(x="country",y = "points",data= df, order =most_country )
f = sns.stripplot(x="country",y = "points",data= df, order =most_country, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot with jitter", loc="left")


# delete features with a lot of NaN
df = df.drop(['region_2','designation'],axis=1)
# replace NaN price by average 
df['price'] = df['price'].fillna(df['price'].mean())
# replace NaN by 'Unknown'
df = df.fillna('Unknown')
f = plt.figure(figsize=(20,5))
sns.heatmap(df.isnull(),yticklabels=False,cmap= 'viridis')