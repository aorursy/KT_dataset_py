print('>> Importing libraries')
import os
import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from kaggle.competitions import twosigmanews
from plotly.graph_objs import Scatter, Figure, Layout
print('>> Data Loading Completed!')

init_notebook_mode(connected=True)
cf.set_config_file(offline=True)
env = twosigmanews.make_env()
market_data, news_data = env.get_training_data()
def mr_inspect(df):
    """Returns a inspection dataframe"""
    print ("Length of dataframe:", len(df))
    inspect_dataframe = pd.DataFrame({'dtype': df.dtypes, 'Unique values': df.nunique() ,
                 'Number of missing values': df.isnull().sum() ,
                  'Percentage missing': (df.isnull().sum() / len(df)) * 100
                 }).sort_values(by='Number of missing values', ascending = False)
    return inspect_dataframe
mr_inspect(market_data)
market_data.head(10)
print ("The oldest date in dataset", market_data['time'].min())
print ("The latest date in dataset", market_data['time'].max())
## Getting random 5 assets by Close Price for trend analysis
random_ten_assets = np.random.choice(market_data['assetName'].unique(), 5)
print(f"Using Top 10 Close Price Assets to Study The Trend:\n{random_ten_assets}")
data = []
for asset in random_ten_assets:
    asset_df = market_data[((market_data['assetName'] == asset) & (market_data['time'].dt.year <= 2009))]
    data.append(go.Bar(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Close Price of Random 5 assets overall for 2007-2009",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))
iplot(dict(data=data, layout=layout), filename='basic-line')
data = []
for asset in random_ten_assets:
    asset_df = market_data[((market_data['assetName'] == asset) & (market_data['time'].dt.year >= 2009))]
    data.append(go.Bar(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Close Price of Random 5 assets overall for 2010-2016",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))
iplot(dict(data=data, layout=layout), filename='basic-line')
market_data['diff'] = market_data['close'] - market_data['open']
groupedByAssetDiff = market_data.groupby('assetCode').agg({'diff': ['std', 'min']}).reset_index()
g = groupedByAssetDiff.sort_values(('diff', 'std'), ascending=False)[:10]
g['min_text'] = 'Maximum price drop: ' + (-1 * g['diff']['min']).astype(str)
trace = go.Scatter(
    x = g['assetCode'].values,
    y = g['diff']['std'].values,
    mode='markers',
    marker=dict(
        size = g['diff']['std'].values,
        color = g['diff']['std'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = g['min_text'].values
    #text = f"Maximum price drop: {g['price_diff']['min'].values}"
    #g['time'].dt.strftime(date_format='%Y-%m-%d').values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Top 10 Assests That Took A Shake!',
    hovermode= 'closest',
    yaxis=dict(
        title= 'diff',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
iplot(dict(data=data, layout=layout), filename='basic-line')
data = []
market_data['month'] = market_data['time'].dt.month
for i in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    price_df = market_data.groupby('month')['returnsOpenNextMktres10'].quantile(i).reset_index()
    data.append(go.Bar(
        x = price_df['month'].values,
        y = price_df['returnsOpenNextMktres10'].values,
        name = f'{i} quantile'
    ))
layout = go.Layout(dict(title = "Trends of grouby Month of returnsOpenNextMktres10 by 10 quartiles ",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="v"),)
iplot(dict(data=data, layout=layout), filename='basic-line')
apple_data = market_data[market_data.assetCode == "AAPL.O"]
apple_data.head()
apple_data[['time','close']].set_index("time").iplot(title='APPLE CLOSE PRICE IN USD', 
                                                     theme='white',color='green', 
                                                     width=6, kind='bar')
market_data.head()
import plotly.plotly as py
import plotly.figure_factory as ff
# Add histogram data
x1 = market_data.returnsClosePrevRaw1[:1000]
x2 = market_data.returnsOpenPrevRaw1[:1000] 

# Group data together
hist_data = [x1, x2]

group_labels = ['returnsClosePrevRaw1 1', 'returnsOpenPrevRaw1 2']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.1)

# Plot!
iplot(fig, filename='Distplot with Multiple Datasets')
import plotly.plotly as py
import plotly.figure_factory as ff
# Add histogram data
x1 = market_data.returnsClosePrevRaw10[:1000]
x2 = market_data.returnsOpenPrevRaw10[:1000] 

# Group data together
hist_data = [x1, x2]
colors = ['#3A4750', '#F64E8B']

group_labels = ['returnsClosePrevRaw10', 'returnsOpenPrevRaw10']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.1, curve_type='normal', colors=colors)

# Plot!
iplot(fig, filename='Distplot with Multiple Datasets')
hist_data = [market_data.returnsOpenNextMktres10.values[:5000]]
group_labels = ['distplot']
fig = ff.create_distplot(hist_data, group_labels)
iplot(fig, filename='Basic Distplot')
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline
plt.figure(figsize=(12,10))
plt.scatter(range(market_data.shape[0]), np.sort(market_data.returnsOpenNextMktres10.values), color='red')
plt.xlabel('index', fontsize=12)
plt.ylabel('returnsOpenNextMktres10', fontsize=12)
plt.show()
#Statistics
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss

#More Viz Libs
import matplotlib.gridspec as gridspec 
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn

#For NLP
import spacy
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   
mr_inspect(news_data)
news_data.head()
alerts_df = news_data[news_data.urgency == 1]
article_df = news_data[news_data.urgency == 3]
eng_stopwords  = list(STOPWORDS)
# alertnews
#wordcloud for alerts comments
text=alerts_df.headline.values[:1000]
wc= WordCloud(background_color="black",max_words=2000,stopwords=eng_stopwords)
wc.generate(" ".join(text))
plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Words frequented in Alert Headlines", fontsize=30)
plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
plt.show()
#wordcloud for articles comments
text=article_df.headline.values[:1000]
wc= WordCloud(background_color="black", max_words=2000, stopwords=eng_stopwords)
wc.generate(" ".join(text))
plt.figure(figsize=(20,10))
plt.axis("off")
plt.title("Words frequented in Article Headlines", fontsize=30)
plt.imshow(wc.recolor(colormap= 'inferno' , random_state=17), alpha=0.98)
plt.show()