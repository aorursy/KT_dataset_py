
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import plotly
plotly.offline.init_notebook_mode(connected=True)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('../input/anime-recommendations-database/anime.csv')
df.shape
df.head()
df.type.unique()
df.genre.unique()
num_romance = len(df[df.genre =='Romance'])
print('There are {} romance anime in the dataset'.format(num_romance))
num_drama = len(df[df.genre =='Drama'])
print('There are {} drama anime in the dataset'.format(num_drama))
mean_rate = df.rating.mean()
print('Mean rating of anime is: {:.2f}'.format(mean_rate))
def top10(df, genre ,type):
    top = df[(df['genre'] == genre) & (df['type'] == type)]
    top_10 = top.head(10)
    return top_10
top10(df, 'Comedy', 'Movie') # CHEBURASHKA HERE!!!
top10(df, 'Drama', 'Movie')
top_rate_ova = df[(df.rating > 9) & (df.type == 'OVA')]
top_rate_ova.head()
top_rate_tv = df[(df.rating > 9) & (df.type == 'TV')]
top_rate_tv.head(10)
top_rate_movie = df[(df.rating > 9) & (df.type == 'Movie')]
top_rate_movie.head(10)
# Let's count anime types in this dataset
sns.set_style('whitegrid')
fig,(ax1) = plt.subplots(figsize=(20,11))
plt.suptitle('Count plots')
sns.countplot(y='type', data=df, ax=ax1)
plt.show()
"""
Second way to count anime types in this dataset
This tool is very handy with solving data visualization

"""
num_of_type_of_anime  = df.type.value_counts().sort_values(ascending=True)
data = [go.Pie(labels = num_of_type_of_anime.index, values = num_of_type_of_anime.values, hoverinfo = 'label+value')]
plotly.offline.iplot(data, filename='active category')
