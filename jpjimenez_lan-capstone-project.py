import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
from wordcloud import WordCloud
#Bring in Two Sigma's New Data
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
stop = set(stopwords.words('english'))

from kaggle.competitions import twosigmanews
# Create Env to save the data
env = twosigmanews.make_env()

# We are going to get our training data
(market_train_df, news_train_df) = env.get_training_data()
def missing_value_graph(data):
    data = [
    go.Bar(
        x = data.columns,
        y = data.isnull().sum(),
        name = 'NULL Fields',
    ),
    ]
    layout= go.Layout(
        title= 'Missing fields',
        xaxis= dict(title='Columns', ticklen=5, zeroline=True, gridwidth=2),
        yaxis=dict(title='Amount of Nulls', ticklen=5, gridwidth=2),
        showlegend=True
    )
    fig= go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='nullList')
missing_value_graph(market_train_df)
missing_value_graph(news_train_df)
def impute_fields(data):
    for x in data.columns:
        if data[x].dtype == "object":
            data[x] = data[x].fillna("filled")
        elif (data[x].dtype == "int64" or data[x].dtype == "float64"):
            data[x] = data[x].fillna(data[x].mean())
        else:
            pass
    return data
market_train_df = impute_fields(market_train_df)
missing_value_graph(market_train_df)
asset_by_volume = market_train_df.groupby("assetCode")["close"].count().to_frame().sort_values(by=['close'],ascending= False)
asset_by_volume = asset_by_volume.sort_values(by=['close'])
top_asset_by_volume = list(asset_by_volume.nlargest(5, ['close']).index)
top_asset_by_volume

for i in top_asset_by_volume:
    asset1_df = market_train_df[(market_train_df['assetCode'] == i) & (market_train_df['time'] > '2007-02-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
    trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['close'].values,
        line = dict(color = 'orange'))

    layout = dict(title = "Closing Price of {}".format(i),
                  xaxis = dict(title = 'Year'),
                  yaxis = dict(title = 'Price in $'),
                  )
    py.iplot(dict(data=[trace1], layout=layout), filename='basic-line')
data = []
for asset in np.random.choice(market_train_df['assetName'].unique(), 10):
    asset_df = market_train_df[(market_train_df['assetName'] == asset)]

    data.append(go.Scatter(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Closing prices of 10 random assets",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))
py.iplot(dict(data=data, layout=layout), filename='basic-line')
text = ' '.join(news_train_df['headline'].str.lower().values[-1000000:])
wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Top words in headline')
plt.show()
for i, j in zip([-1, 0, 1], ['negative', 'neutral', 'positive']):
    df_sentiment = news_train_df.loc[news_train_df['sentimentClass'] == i, 'assetName']
    print(f'Top mentioned companies for {j} sentiment are:')
    print(df_sentiment.value_counts().head(5))
    print('')
(news_train_df['headlineTag'].value_counts() / 1000)[:10].plot('barh');
plt.title('headlineTag counts (thousands)');