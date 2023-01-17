import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

# from plotly.tools import FigureFactory as FF 

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

from kaggle.competitions import twosigmanews

# You can only call make_env() once, so don't lose it!

env = twosigmanews.make_env()

print('Done!')
(market_train_df, news_train_df) = env.get_training_data()
market_train, news_train = market_train_df, news_train_df
import pandas as pd

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

%matplotlib inline

import seaborn as sns

import numpy as np

import plotly.figure_factory as ff





######### Function

def mis_value_graph(data):

#     data.isnull().sum().plot(kind="bar", figsize = (20,10), fontsize = 20)

#     plt.xlabel("Columns", fontsize = 20)

#     plt.ylabel("Value Count", fontsize = 20)

#     plt.title("Total Missing Value By Column", fontsize = 20)

#     for i in range(len(data)):

#          colors.append(generate_color())

            

    data = [

    go.Bar(

        x = data.columns,

        y = data.isnull().sum(),

        name = 'Unknown Assets',

        textfont=dict(size=20),

        marker=dict(

#         color= colors,

        line=dict(

            color='#000000',

            width=2,

        ), opacity = 0.45

    )

    ),

    ]

    layout= go.Layout(

        title= '"Total Missing Value By Column"',

        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),

        yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),

        showlegend=True

    )

    fig= go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='skin')

    



def mis_impute(data):

    for i in data.columns:

        if data[i].dtype == "object":

            data[i] = data[i].fillna("other")

        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):

            data[i] = data[i].fillna(data[i].mean())

        else:

            pass

    return data





import random



def generate_color():

    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))

    return color
mis_value_graph(market_train_df)

market_train_df = mis_impute(market_train_df)

market_train_df.isna().sum().to_frame()
# https://www.kaggle.com/pestipeti/simple-eda-two-sigma

best_asset_volume = market_train_df.groupby("assetCode")["close"].count().to_frame().sort_values(by=['close'],ascending= False)

best_asset_volume = best_asset_volume.sort_values(by=['close'])

largest_by_volume = list(best_asset_volume.nlargest(10, ['close']).index)

# largest_by_volume

for i in largest_by_volume:

    asset1_df = market_train_df[(market_train_df['assetCode'] == i) & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]

    # Create a trace

    trace1 = go.Scatter(

        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,

        y = asset1_df['close'].values,

        line = dict(color = generate_color()),opacity = 0.8

    )



    layout = dict(title = "Closing prices of {}".format(i),

                  xaxis = dict(title = 'Month'),

                  yaxis = dict(title = 'Price (USD)'),

                  )



    data = [trace1]

    py.iplot(dict(data=data, layout=layout), filename='basic-line')
for i in largest_by_volume:



    asset1_df['high'] = asset1_df['open']

    asset1_df['low'] = asset1_df['close']



    for ind, row in asset1_df.iterrows():

        if row['close'] > row['open']:

            

            asset1_df.loc[ind, 'high'] = row['close']

            asset1_df.loc[ind, 'low'] = row['open']



    trace1 = go.Candlestick(

        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,

        open = asset1_df['open'].values,

        low = asset1_df['low'].values,

        high = asset1_df['high'].values,

        close = asset1_df['close'].values,

        increasing=dict(line=dict(color= generate_color())),

        decreasing=dict(line=dict(color= generate_color())))



    layout = dict(title = "Candlestick chart for {}".format(i),

                  xaxis = dict(

                      title = 'Month',

                      rangeslider = dict(visible = False)

                  ),

                  yaxis = dict(title = 'Price (USD)')

                 )

    data = [trace1]



    py.iplot(dict(data=data, layout=layout), filename='basic-line')
for i in range(1,100,10):

    volumeByAssets = market_train_df.groupby(market_train_df['assetCode'])['volume'].sum()

    highestVolumes = volumeByAssets.sort_values(ascending=False)[i:i+9]

    # Create a trace

    colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

    trace1 = go.Pie(

        labels = highestVolumes.index,

        values = highestVolumes.values,

        textfont=dict(size=20),

        marker=dict(colors=colors,line=dict(color='#000000', width=2)), hole = 0.45)

    layout = dict(title = "Highest trading volumes for range of {} to {}".format(i, i+9))

    data = [trace1]

    py.iplot(dict(data=data, layout=layout), filename='basic-line')
assetNameGB = market_train_df[market_train_df['assetName'] == 'Unknown'].groupby('assetCode')

unknownAssets = assetNameGB.size().reset_index('assetCode')

unknownAssets.columns = ['assetCode',"value"]

unknownAssets = unknownAssets.sort_values("value", ascending= False)

unknownAssets.head(5)



colors = []

for i in range(len(unknownAssets)):

     colors.append(generate_color())



        

data = [

    go.Bar(

        x = unknownAssets.assetCode.head(25),

        y = unknownAssets.value.head(25),

        name = 'Unknown Assets',

        textfont=dict(size=20),

        marker=dict(

        color= colors,

        line=dict(

            color='#000000',

            width=2,

        ), opacity = 0.45

    )

    ),

    ]

layout= go.Layout(

    title= 'Unknown Assets by Asset code',

    xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),

    yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),

    showlegend=True

)

fig= go.Figure(data=data, layout=layout)

py.iplot(fig, filename='skin')
mis_value_graph(news_train_df)

news_train_df = mis_impute(news_train_df)

news_train_df.isna().sum().to_frame()
print("News data shape",news_train_df.shape)

news_train_df.head()
# news_train_df['urgency'].value_counts()

news_sentiment_count = news_train_df.groupby(["urgency","assetName"])[["sentimentNegative","sentimentNeutral","sentimentPositive"]].count()

news_sentiment_count = news_sentiment_count.reset_index()
trace = go.Table(

    header=dict(values=list(news_sentiment_count.columns),

                fill = dict(color='rgba(55, 128, 191, 0.7)'),

                align = ['left'] * 5),

    cells=dict(values=[news_sentiment_count.urgency,news_sentiment_count.assetName,news_sentiment_count["sentimentNegative"], news_sentiment_count["sentimentPositive"], news_sentiment_count["sentimentNeutral"]],

               fill = dict(color='rgba(245, 246, 249, 1)'),

               align = ['left'] * 5))



data = [trace] 

py.iplot(data, filename = 'pandas_table')
trace0 = go.Bar(

    x= news_sentiment_count.assetName.head(30),

    y=news_sentiment_count.sentimentNegative.values,

    name='sentimentNegative',

    textfont=dict(size=20),

        marker=dict(

        color= generate_color(),

        opacity = 0.87

    )

)

trace1 = go.Bar(

    x= news_sentiment_count.assetName.head(30),

    y=news_sentiment_count.sentimentNeutral.values,

    name='sentimentNeutral',

    textfont=dict(size=20),

        marker=dict(

        color= generate_color(),

        opacity = 0.87

    )

)

trace2 = go.Bar(

    x= news_sentiment_count.assetName.head(30),

    y=news_sentiment_count.sentimentPositive.values,

    name='sentimentPositive',

    textfont=dict(size=20),

    marker=dict(

        color= generate_color(),

        opacity = 0.87

    )

)



data = [trace0, trace1, trace2]

layout = go.Layout(

    xaxis=dict(tickangle=-45),

    barmode='group',

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='angled-text-bar')
news_sentiment_urgency = news_train_df.groupby(["urgency"])[["sentimentNegative","sentimentNeutral","sentimentPositive"]].count()

news_sentiment_urgency = news_sentiment_urgency.reset_index()
trace = go.Table(

    header=dict(values=list(news_sentiment_urgency.columns),

                fill = dict(color='rgba(55, 128, 191, 0.7)'),

                align = ['left'] * 5),

    cells=dict(values=[news_sentiment_urgency.urgency,news_sentiment_urgency["sentimentNegative"], news_sentiment_urgency["sentimentPositive"], news_sentiment_urgency["sentimentNeutral"]],

               fill = dict(color='rgba(245, 246, 249, 1)'),

               align = ['left'] * 5))



data = [trace] 

py.iplot(data, filename = 'pandas_table')
trace0 = go.Bar(

    x= news_sentiment_urgency.urgency.values,

    y=news_sentiment_urgency.sentimentNegative.values,

    name='sentimentNegative',

    textfont=dict(size=20),

        marker=dict(

        color= generate_color(),

            line=dict(

            color='#000000',

            width=2,

        ),

        opacity = 0.87

    )

)

trace1 = go.Bar(

    x= news_sentiment_urgency.urgency.values,

    y=news_sentiment_urgency.sentimentNegative.values,

    name='sentimentNeutral',

    textfont=dict(size=20),

        marker=dict(

        color= generate_color(),

        line=dict(

            color='#000000',

            width=2,

        ),

        opacity = 0.87

    )

)

trace2 = go.Bar(

    x= news_sentiment_urgency.urgency.values,

    y=news_sentiment_urgency.sentimentNegative.values,

    name='sentimentPositive',

    textfont=dict(size=20),

    marker=dict(

        line=dict(

            color='#000000',

            width=2,

        ),

        color= generate_color(),

        opacity = 0.87

    )

)

data = [trace0, trace1, trace2]

layout = go.Layout(

    xaxis=dict(tickangle=-45),

    barmode='group',

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='angled-text-bar')
%%time

def data_prep(market_train,news_train):

    market_train.time = market_train.time.dt.date

    news_train.time = news_train.time.dt.hour

    news_train.sourceTimestamp= news_train.sourceTimestamp.dt.hour

    news_train.firstCreated = news_train.firstCreated.dt.date

    news_train['assetCodesLen'] = news_train['assetCodes'].map(lambda x: len(eval(x)))

    news_train['assetCodes'] = news_train['assetCodes'].map(lambda x: list(eval(x))[0])

    kcol = ['firstCreated', 'assetCodes']

    news_train = news_train.groupby(kcol, as_index=False).mean()

    market_train = pd.merge(market_train, news_train, how='left', left_on=['time', 'assetCode'], 

                            right_on=['firstCreated', 'assetCodes'])

    lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}

    market_train['assetCodeT'] = market_train['assetCode'].map(lbl)

    

    

    market_train = market_train.dropna(axis=0)

    

    return market_train



market_train = data_prep(market_train_df, news_train_df)

market_train.shape
%%time

# The target is binary

up = market_train.returnsOpenNextMktres10 >= 0



fcol = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 

                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 

                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]
%%time

from xgboost import XGBClassifier

from sklearn import model_selection

from sklearn.metrics import accuracy_score



X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.25, random_state=99)



xgb_up = XGBClassifier(n_jobs=4,n_estimators=200,max_depth=8,eta=0.1)

print('Fitting Up')

xgb_up.fit(X_train,up_train)

print("Accuracy Score: ",accuracy_score(xgb_up.predict(X_test),up_test))
import matplotlib.pyplot as plt

import seaborn as sns



df = pd.DataFrame({'imp': xgb_up.feature_importances_, 'col':fcol})

df = df.sort_values(['imp','col'], ascending=[True, False])

# _ = df.plot(kind='barh', x='col', y='imp', figsize=(7,12))





#plt.savefig('lgb_gain.png')

trace = go.Table(

    header=dict(values=list(df.columns),

                fill = dict(color='rgba(55, 128, 191, 0.7)'),

                align = ['left'] * 5),

    cells=dict(values=[df.imp,df.col],

               fill = dict(color='rgba(245, 246, 249, 1)'),

               align = ['left'] * 5))



data = [trace] 

py.iplot(data, filename = 'pandas_table')