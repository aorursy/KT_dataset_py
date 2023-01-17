#Baseado no Kernel https://www.kaggle.com/artgor/eda-feature-engineering-and-everything/notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
stop = set(stopwords.words('english'))


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head()
news_train_df.head()
#Tamanho dos DataFrame
print(f'{market_train_df.shape[0]} registros e {market_train_df.shape[1]} colunas no conjunto de treinamento.')
#Tipo dos dados de cada coluna
market_train_df.dtypes
#Quantidade de valores ausentes(NaN) por coluna
market_train_df.isna().sum()
#Descrição de "assetCode"
market_train_df['assetCode'].describe()
#Descrição de "assetName"
market_train_df['assetName'].describe()
#Gráfico da flutuação dos preços de algumas empresas ao longo de um periodo 

data = []
for asset in np.random.choice(market_train_df['assetName'].unique(), 10):
    asset_df = market_train_df[(market_train_df['assetName'] == asset)]

    data.append(go.Scatter(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Preços de fechamento de 10 ativos aleatórios",
                  xaxis = dict(title = 'Mês'),
                  yaxis = dict(title = 'Preço (close)'),
                  ),legend=dict(
                orientation="v"))
py.iplot(dict(data=data, layout=layout), filename='basic-line')
#Gráfico do periodo com maior flutuaçao dos preços

market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']
grouped = market_train_df.groupby('time').agg({'price_diff': ['std', 'min']}).reset_index()

g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:10]
g['min_text'] = 'Máxima queda de preço: ' + (-1 * g['price_diff']['min']).astype(str)

trace = go.Scatter(
    x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = -1*g['price_diff']['min'].values,
    mode='markers',
    marker=dict(size = g['price_diff']['std'].values,color=g['price_diff']['std'].values,
                colorscale='Portland',showscale=True),
    text = g['min_text'].values
    )

data = [trace]
layout= go.Layout(autosize= True,title='10 meses com maiores flutuações',hovermode='closest',
    yaxis=dict(title='diferença de preço (close-open)', ticklen=5, gridwidth=2),
    showlegend= False)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')
#Pegar os 10 maiores valores de coluna price_diff
market_train_df.sort_values('price_diff')[:10]
#Visualizar a flutuação dos preços da IBM

asset1Code = 'IBM.N'
asset1_df = market_train_df[(market_train_df['assetCode'] == asset1Code)]

trace1 = go.Scatter(
    x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = asset1_df['price_diff'].values
)

layout = dict(title = "Diferença de preço no fechamento e na abertura da {}".format(asset1Code),
              xaxis = dict(title = 'Mês'),
              yaxis = dict(title = 'Diferença de preço (close-open)'),
              )
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')
market_train_df['close_to_open'] =  np.abs(market_train_df['close'] / market_train_df['open'])


print(f"Em {(market_train_df['close_to_open'] >= 2).sum() + (market_train_df['close_to_open'] <= 0.5).sum()} casos o preço aumentou ou diminuiu em 100% ou mais em um dia")
#print(f"In {(market_train_df['close_to_open'] <= 0.5).sum()} lines price decreased by 100% or more.")
news_train_df.head()
print(f'{news_train_df.shape[0]} registros e {news_train_df.shape[1]} colunas no conjunto de treinamento.')
#Nuvem de palavras (WordCloud)
text = ' '.join(news_train_df['headline'].str.lower().values[-100000:])
wordcloud = WordCloud(max_font_size=None, stopwords=stop, background_color='white',
                      width=1200, height=1000).generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud)
plt.title('Principais palavras nos títulos')
plt.axis("off")
plt.show()
#Quantidade de palavras por frase

news_train_df['sentence_word_count'] =  news_train_df['wordCount'] / news_train_df['sentenceCount']
plt.boxplot(news_train_df['sentence_word_count'][news_train_df['sentence_word_count'] < 40]);
plt.ylabel('Quantidade de palavras por frase')
#Organizações que forneceram o maior número de notícias
news_train_df['provider'].value_counts().head(10)
#Tags da manchete 

(news_train_df['headlineTag'].value_counts() / 1000)[:10].plot('barh');
plt.title('Tags da manchete (headlineTag)');
#Empresas com o maior número de sitacões positivas, negativas e neutras

for i, j in zip([-1, 0, 1], ['negativo', 'neutro', 'positivo']):
    df_sentiment = news_train_df.loc[news_train_df['sentimentClass'] == i, 'assetName']
    print(f'As principais empresas mencionadas de forma {j} são:')
    print(df_sentiment.value_counts().head(5))
    print('')
def data_prep(market_df,news_df):
    market_df['time'] = market_df.time.dt.date
    market_df['returnsOpenPrevRaw1_to_volume'] = market_df['returnsOpenPrevRaw1'] / market_df['volume']
    market_df['close_to_open'] = market_df['close'] / market_df['open']
    market_df['volume_to_mean'] = market_df['volume'] / market_df['volume'].mean()
    
    news_df['time'] = news_df.time.dt.hour
    news_df['sourceTimestamp']= news_df.sourceTimestamp.dt.hour
    news_df['firstCreated'] = news_df.firstCreated.dt.date
    news_df['assetCodesLen'] = news_df['assetCodes'].map(lambda x: len(eval(x)))
    news_df['assetCodes'] = news_df['assetCodes'].map(lambda x: list(eval(x))[0])
    news_df['headlineLen'] = news_df['headline'].apply(lambda x: len(x))
    news_df['assetCodesLen'] = news_df['assetCodes'].apply(lambda x: len(x))
    news_df['asset_sentiment_count'] = news_df.groupby(['assetName', 'sentimentClass'])['time'].transform('count')
    news_df['asset_sentence_mean'] = news_df.groupby(['assetName', 'sentenceCount'])['time'].transform('mean')
    lbl = {k: v for v, k in enumerate(news_df['headlineTag'].unique())}
    news_df['headlineTagT'] = news_df['headlineTag'].map(lbl)
    kcol = ['firstCreated', 'assetCodes']
    news_df = news_df.groupby(kcol, as_index=False).mean()

    market_df = pd.merge(market_df, news_df, how='left', left_on=['time', 'assetCode'], 
                            right_on=['firstCreated', 'assetCodes'])

    lbl = {k: v for v, k in enumerate(market_df['assetCode'].unique())}
    market_df['assetCodeT'] = market_df['assetCode'].map(lbl)
    
    market_df = market_df.dropna(axis=0)
    
    return market_df


news_train_df = news_train_df.loc[news_train_df['time'] >= '2010-01-01 22:00:00+0000']
market_train_df.drop(['price_diff', 'close_to_open'], axis=1, inplace=True)
market_train = data_prep(market_train_df, news_train_df)
print(market_train.shape)
up = market_train.returnsOpenNextMktres10 >= 0

fcol = [c for c in market_train.columns if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'assetCodeT', 'volume_to_mean', 'sentence_word_count',
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 'returnsOpenPrevRaw1_to_volume',
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]

X = market_train[fcol].values
up = up.values
r = market_train.returnsOpenNextMktres10.values

# Scaling of X values
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)
X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.1, random_state=99)

xgb_up = XGBClassifier(n_estimators=300,
                       max_depth=4,
                       eta=0.2,
                       random_state=10)
%%time
xgb_up.fit(X_train,up_train)
print("Accuracy Score: ",accuracy_score(xgb_up.predict(X_test),up_test))
days = env.get_prediction_days()
import time

n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    if n_days % 50 == 0:
        print(n_days,end=' ')
    
    t = time.time()
    market_obs_df = data_prep(market_obs_df, news_obs_df)
    market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    X_live = market_obs_df[fcol].values
    X_live = 1 - ((maxs - X_live) / rng)
    prep_time += time.time() - t
    
    t = time.time()
    lp = xgb_up.predict_proba(X_live)
    prediction_time += time.time() -t
    
    t = time.time()
    confidence = 2* lp[:,1] -1
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t
    

env.write_submission_file()
