# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np 

import pandas as pd

import os

import seaborn as sns

print(os.listdir("../input"))

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=False)

from wordcloud import WordCloud

from geopy.geocoders import Nominatim

from folium.plugins import HeatMap

import folium

from tqdm import tqdm

import re

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

import gensim

from collections import Counter

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import matplotlib.colors as mcolors

from sklearn.manifold import TSNE

from gensim.models import word2vec

import nltk

import os

print(os.listdir("../input"))
df=pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')

df.head()
plt.figure(figsize=(10,7))

chain=df['name'].value_counts()[0:20]

sns.barplot(x=chain,y=chain.index)

plt.xlabel('no. of outlets')
x=df['online_order'].value_counts()



plt.pie(x,labels=x.index, autopct='%.0f%%', shadow=True )

plt.title('Accepting vs not accepting online orders')

plt.show()
y=df['book_table'].value_counts()

plt.pie(y,labels=y.index,autopct='%.0f%%',shadow=True,colors=['#96D38C', '#D0F9B1'])
rating=df['rate'].dropna().apply(lambda x:float(x.split('/')[0])if (len(x)>3)  else np.nan ).dropna()
sns.distplot(rating,bins=20)

plt.show()
cost_dist=df[['rate','approx_cost(for two people)','online_order']].dropna()

cost_dist['cost_dist']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)

cost_dist['approx_cost']=cost_dist['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))
plt.figure(figsize=(10,7))

sns.scatterplot(x='cost_dist',y='approx_cost',hue='online_order',data=cost_dist)
plt.figure(figsize=(6,6))

sns.distplot(cost_dist['approx_cost'])

plt.show()
plt.figure(figsize=(10,7))

rest=df['rest_type'].value_counts()[:20]

sns.barplot(rest,rest.index)

plt.title("Restaurant types")

plt.xlabel("count")
plt.figure(figsize=(10,7))

rest=df['listed_in(city)'].value_counts()[:20]

sns.barplot(rest,rest.index)
plt.figure(figsize=(10,7))

cui=df['cuisines'].value_counts()[:10]

sns.barplot(cui,cui.index)

plt.xlabel('COUNT')

plt.show()
df.head()
plt.figure(figsize=(10,7))

rates=cost_dist['cost_dist'].value_counts()

sns.barplot(x=rates.index,y=rates)
import statistics 

x=statistics.stdev(rates.index)

y=statistics.mean(rates.index) 

print(f'Ratings follows Normal distribution with mean {x} and standard deviation {y}')