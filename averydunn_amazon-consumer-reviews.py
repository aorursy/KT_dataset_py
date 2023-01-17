# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
reviews = pd.read_csv('/kaggle/input/consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')
reviews
reviews.loc[:, ['reviews.text', 'reviews.title']]
!pip install vaderSentiment 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
analyzer = SentimentIntensityAnalyzer()
df = pd.DataFrame(([analyzer.polarity_scores(text)['neg']] for text in reviews['reviews.text']), columns=['negative'])
df['positive'] = [(analyzer.polarity_scores(text)['pos']) for text in reviews['reviews.text']]
df['neutral'] = [(analyzer.polarity_scores(text)['neu']) for text in reviews['reviews.text']]
df['compound'] = [(analyzer.polarity_scores(text)['compound']) for text in reviews['reviews.text']]

#joining dataframe with the title of reviews 
desc = reviews['reviews.title']
df = df.join(desc)

rate = reviews['reviews.rating']
df = df.join(rate)

df.head(50)
# these are advanced visualization tools, modeled off another notebook
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff
ratings = reviews['reviews.rating'].value_counts()

label_rating = ratings.index
size_rating = ratings.values

fig1, ax1 = plt.subplots()
ax1.pie(size_rating, labels=label_rating)
ax1.axis('equal')
plt.show()
ratings = reviews['reviews.rating'].value_counts()

label_rating = ratings.index
size_rating = ratings.values

colors = ['orange', 'green', 'purple', 'yellow', 'red']

rating_piechart = go.Pie(labels = label_rating,
                         values = size_rating,
                         marker = dict(colors = colors),
                         name = 'Ratings', hole = 0.1)

df = [rating_piechart]

layout = go.Layout(
           title = 'Distribution of Ratings of Amazon products')

fig = go.Figure(data = df,
                 layout = layout)

py.iplot(fig)