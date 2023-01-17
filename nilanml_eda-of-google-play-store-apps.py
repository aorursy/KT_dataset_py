import pandas as pd
from importlib import reload
import sys
from imp import reload
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('../input/googleplaystore.csv')
df.head()
df.Price.unique()
df.Price = df.Price.apply(lambda x: str(x).replace("$",""))
df.Price.unique()
print(df.shape)
df = df.drop_duplicates(subset=['App'], keep = 'first')
print(df.shape)
df.dtypes
df.Reviews = pd.to_numeric(df.Reviews, errors='coerce')
df.Price = pd.to_numeric(df.Price, errors='coerce')
df.Rating = pd.to_numeric(df.Rating, errors='coerce')
df.dtypes
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
fig = sns.countplot(x=df['Installs'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show(fig)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(5,3))
fig = sns.countplot(x=df['Type'])
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show(fig)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6,3))
fig = sns.countplot(x=df['Content Rating'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show(fig)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
fig = sns.countplot(x=df['Category'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show(fig)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(30,5))
fig = sns.countplot(x=df['Genres'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show(fig)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,3))
fig = sns.barplot(y=df['Genres'].value_counts().reset_index()[:10]['Genres'], x=df['Genres'].value_counts().reset_index()[:10]['index'], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.show(fig)
sorted_by_rating = df.sort_values(by=['Rating'], ascending=False)
sorted_by_rating.head()
plt.figure(figsize=(8,6))
fig = sns.barplot(x=sorted_by_rating['App'][:10], y=sorted_by_rating['Rating'][:10], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.show(fig)
sorted_by_reviews = df.sort_values(by=['Reviews'], ascending=False)
sorted_by_reviews.head()
plt.figure(figsize=(8,6))
fig = sns.barplot(x=sorted_by_reviews['App'][:10], y=sorted_by_reviews['Reviews'][:10], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.show(fig)
sorted_by_price = df.sort_values(by=['Price'], ascending=False)
sorted_by_price.head()
plt.figure(figsize=(8,6))
fig = sns.barplot(x=sorted_by_price['App'][:10], y=sorted_by_price['Price'][:10], palette="hls")
fig.set_xticklabels(fig.get_xticklabels(),rotation=90)
plt.tight_layout()
plt.show(fig)
