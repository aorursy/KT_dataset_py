import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews

# Interactive graphs
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()
#market_train.loc[market_train['assetCode'] == 'AAPL.O', ['assetCode']]
data = []
df = market_train[(market_train['assetCode'] == 'AAPL.O')]

data.append(go.Scatter(
    x = df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = df['close'].values,
    name = 'APPL.O'
))

layout = go.Layout(dict(title = "Closing prices of Apple",
                       xaxis = dict(title = 'Year'),
                       yaxis = dict(title = 'Price (USD)'),
                       ), legend = dict(
                        orientation='h'))
py.iplot(dict(data=data, layout=layout), filename='basic-line')
data = []
df = df.loc[pd.to_datetime(df['time']) >= pd.to_datetime('2014-06-09').tz_localize('UTC')]

data.append(go.Scatter(
    x = df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = df['close'].values,
    name = 'APPL.O'
))

layout = go.Layout(dict(title = "Closing prices of Apple",
                       xaxis = dict(title = 'Year'),
                       yaxis = dict(title = 'Price (USD)'),
                       ), legend = dict(
                        orientation='h'))
py.iplot(dict(data=data, layout=layout), filename='basic-line')
news_df = news_train[(news_train['assetName'] == 'Apple Inc')]
news_df = news_df.loc[pd.to_datetime(news_df['time']) >= pd.to_datetime('2014-06-09').tz_localize('UTC')]

# time = pd.to_datetime('2014-06-09').tz_localize('UTC')
# time += pd.Timedelta(days=1)
# time
    
start_day = pd.to_datetime('2014-06-09').tz_localize('UTC')
day_after = pd.to_datetime('2014-06-10').tz_localize('UTC')
sent_count = 0
sent_neg_total = 0
sent_neut_total = 0
sent_pos_total = 0
sent_neg = []
sent_neut = []
sent_pos = []
dates = []

# Function that calculates the average sentiment for each day
for index, row in news_df.iterrows():
    if pd.to_datetime(row['time']) > start_day and pd.to_datetime(row['time']) < day_after:
        sent_count += 1
        sent_neg_total += row['sentimentNegative']
        sent_neut_total += row['sentimentNeutral']
        sent_pos_total += row['sentimentPositive']
    else:
        if sent_count == 0:
            sent_count = 1
        sent_neg.append(sent_neg_total/sent_count)
        sent_neut.append(sent_neut_total/sent_count)
        sent_pos.append(sent_pos_total/sent_count)
        sent_count = 0
        sent_neg_total = 0
        sent_neut_total = 0
        sent_pos_total = 0
        dates.append(start_day)
        start_day += pd.Timedelta(days=1)
        day_after += pd.Timedelta(days=1)

# Plot average sentiment values for each day
neg_sent = go.Scatter(
    x = dates,
    y = sent_neg,
    mode = 'markers',
    name = 'Negative Sentiment' 
)

neut_sent = go.Scatter(
    x = dates,
    y = sent_neut,
    mode = 'markers',
    name = 'Neutral Sentiment' 
)

pos_sent = go.Scatter(
    x = dates,
    y = sent_pos,
    mode = 'markers',
    name = 'Positive Sentiment' 
)

data = [neg_sent, neut_sent, pos_sent]

layout = go.Layout(dict(title = "Sentiment of Apple",
                       xaxis = dict(title = 'Year'),
                       yaxis = dict(title = 'Sentiment Value'),
                       ), legend = dict(
                        orientation='h'))
py.iplot(dict(data=data, layout=layout), filename='line-mode')


######################################################################################
# neg_sent = go.Scatter(
#     x = news_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
#     y = news_df['sentimentNegative'].values,
#     mode = 'markers',
#     name = 'Negative Sentiment' 
# )

# neut_sent = go.Scatter(
#     x = news_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
#     y = news_df['sentimentNeutral'].values,
#     mode = 'markers',
#     name = 'Neutral Sentiment' 
# )

# pos_sent = go.Scatter(
#     x = news_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
#     y = news_df['sentimentPositive'].values,
#     mode = 'markers',
#     name = 'Positive Sentiment' 
# )

# data = [neg_sent, neut_sent, pos_sent]

# layout = go.Layout(dict(title = "Sentiment of Apple",
#                        xaxis = dict(title = 'Year'),
#                        yaxis = dict(title = 'Sentiment Value'),
#                        ), legend = dict(
#                         orientation='h'))
# py.iplot(dict(data=data, layout=layout), filename='line-mode')
news_df = news_train[(news_train['assetName'] == 'Apple Inc')]
news_df = news_df.loc[pd.to_datetime(news_df['time']) >= pd.to_datetime('2014-06-09').tz_localize('UTC')]

# time = pd.to_datetime('2014-06-09').tz_localize('UTC')
# time += pd.Timedelta(days=1)
# time
    
start_day = pd.to_datetime('2014-06-09').tz_localize('UTC')
day_after = pd.to_datetime('2014-06-10').tz_localize('UTC')
sent_count = 0
sent_neg_total = 0
sent_neut_total = 0
sent_pos_total = 0
sent_overall = []
dates = []

# Function that calculates the average sentiment for each day
for index, row in news_df.iterrows():
    if pd.to_datetime(row['time']) > start_day and pd.to_datetime(row['time']) < day_after:
        sent_count += 1
        sent_neg_total += row['sentimentNegative']
        sent_neut_total += row['sentimentNeutral']
        sent_pos_total += row['sentimentPositive']
    else:
        if sent_count == 0:
            start_day += pd.Timedelta(days=1)
            day_after += pd.Timedelta(days=1)
            sent_count = 0
            sent_neg_total = 0
            sent_neut_total = 0
            sent_pos_total = 0
        else:
            sent_neg_total = sent_neg_total/sent_count
            sent_neut_total = sent_neut_total/sent_count
            sent_pos_total = sent_pos_total/sent_count
            sent_overall.append(sent_pos_total - sent_neg_total)
            sent_count = 0
            sent_neg_total = 0
            sent_neut_total = 0
            sent_pos_total = 0
            dates.append(start_day)
            start_day += pd.Timedelta(days=1)
            day_after += pd.Timedelta(days=1)

# Plot average sentiment values for each day
overall_sentiment = go.Scatter(
    x = dates,
    y = sent_overall,
#     mode = 'markers',
    name = 'Overall Sentiment Score' 
)

data = [overall_sentiment]

layout = go.Layout(dict(title = "Sentiment of Apple",
                       xaxis = dict(title = 'Year'),
                       yaxis = dict(title = 'Sentiment Value'),
                       ), legend = dict(
                        orientation='h'))
py.iplot(dict(data=data, layout=layout), filename='line-mode')
apple_close_price = (go.Scatter(
    x = df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = df['close'].values,
    name = 'APPL.O'
))

data = [overall_sentiment, apple_close_price]

layout = go.Layout(dict(title = "Sentiment of Apple",
                       xaxis = dict(title = 'Year'),
                       yaxis = dict(title = 'Sentiment Value'),
                       ), legend = dict(
                        orientation='h'))
py.iplot(dict(data=data, layout=layout), filename='line-mode')
market_train.loc[market_train['assetCode'] == 'A.N', ['time', 'assetCode']]

target_col = ['returnsOpenNextMktres10']
cat_cols = ['assetCode']
num_cols = ['volume', 'close', 'open',
       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
from sklearn.model_selection import train_test_split

market_train = market_train.loc[pd.to_datetime(market_train['time']) >= pd.to_datetime('2009-01-01').tz_localize('UTC')]

train_indices, val_indices = train_test_split(market_train.index.values, test_size = 0.25, random_state = 23)
# Handles categorical variables

def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

encoders = [{} for cat in cat_cols]

for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end = ' ')
    encoders[i] = {l: id for id, l in enumerate(market_train.loc[train_indices, cat].astype(str).unique())}
    market_train[cat] = market_train[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done')
    
embed_sizes = [len(encoder) + 1 for encoder in encoders]
encoders
# Handles numerical variables
from sklearn.preprocessing import StandardScaler
from datetime import datetime

market_train[num_cols] = market_train[num_cols].fillna(0)
print('scaling numerical columns')

scaler = StandardScaler()
print(market_train['time'].dtypes)
market_train[num_cols] = scaler.fit_transform(market_train[num_cols])
# Prepare data and get variables to calculate scoring metric
def get_input(market_train, indices):
    X_num = market_train.loc[indices, num_cols].values
    X = {'num': X_num}
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat_cols].values
    y = (market_train.loc[indices, 'returnsOpenNextMktres10'] >= 0).values
    r = market_train.loc[indices, 'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time'].dt.date
    return X, y, r, u, d # r, u, and d are used to calculate the scoring metric

X_train, y_train, r_train, u_train, d_train = get_input(market_train, train_indices)
X_valid, y_valid, r_valid, u_valid, d_valid = get_input(market_train, val_indices)
# Magic XG Boost Model
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings(action='ignore',category=DeprecationWarning)

model = XGBClassifier(n_jobs = 4, n_estimators = 47, max_depth = 6)
model.fit(X_train['num'], y_train.astype(int))
confidence_valid = model.predict(X_valid['num'])*2-1
score = accuracy_score(confidence_valid>0, y_valid)
print(score)
# Calculation of actual metric that is used for final score
r_valid = r_valid.clip(-1,1) # get rid out outliers
x_t_i = confidence_valid * r_valid * u_valid
data = {'day': d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)
r_valid
x_t_i
confidence_valid
y_valid
plt.hist(confidence_valid, bins = 'auto')
plt.title("predicted confidence")
plt.show()