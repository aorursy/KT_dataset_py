# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')
(market_train_df, news_train_df) = env.get_training_data()
market_train, news_train = market_train_df, news_train_df
%%time
# The target is binary
target = market_train.returnsOpenNextMktres10

fcol_market = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]
fcol_news = [c for c in news_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]
print(plt.style.available)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%matplotlib inline
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff

######### Function
def mis_value_graph(data):

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
    
mis_value_graph(market_train_df)
#Questa funzione è equivalente a quella fatta con il Simple Imputer
def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data

market_train_with_imputer = mis_impute(market_train_df)
#conteggio di NAN rimasti presente => devono essere zero!
market_train_with_imputer.isna().sum().to_frame()
#verifica delle volte che è presente A.N
AN=market_train_with_imputer['assetCode'].where(market_train_with_imputer['assetCode'] == 'A.N').count()
AN
best_asset_volume = market_train_df.groupby("assetCode")["close"]  #Raggruppa le due colonne in base alle volte che è presente uno stesso AssetCode
best_asset_volume=best_asset_volume.count()          #Conteggia quante volte è presente AssetCode
best_asset_volume=best_asset_volume.to_frame()      #Diventa un dataframe
best_asset_volume=best_asset_volume.sort_values(by=['close'],ascending= False)   #ordina in maniera discendente

#best_asset_volume_asc_close = best_asset_volume.sort_values(by=['close'])
largest_by_volume = list(best_asset_volume.nlargest(10, ['close']).index)
largest_by_volume
#best_asset_volume.head(10)
for i in largest_by_volume:
    asset1_df = market_train_df[(market_train_df['assetCode'] == i) & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
    trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['close'].values,
        line = dict(color = 'rgb(1,100,100)')
    )

    layout = go.Layout(
                  title = "Closing prices of {}".format(i),
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )

    data = [trace1]
    fig=go.Figure(data=data,layout=layout)
    py.iplot(fig, filename='basic-line')
for i in largest_by_volume:
    asset1_df = market_train_df[(market_train_df['assetCode'] == i) & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
    trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['close'].values,
        line = dict(color = 'rgb(1,100,100)')
    )
    trace2 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['open'].values,
        line = dict(color = 'rgb(100,1,100)')
    )

    layout = go.Layout(
                  title = "Closing prices of {}".format(i),
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )

    data = [trace1,trace2]
    fig=go.Figure(data=data,layout=layout)
    py.iplot(fig, filename='basic-line')
asset_time = market_train_df.groupby("time")["assetCode"]  #Raggruppa le due colonne in base alle volte che è presente uno stesso AssetCode
asset_time=asset_time.count()          #Conteggia quante volte è presente AssetCode
asset_time=asset_time.to_frame()      #Diventa un dataframe
asset_time=asset_time.sort_values(by=['time'],ascending= False)   #ordina in maniera discendente
#asset_time.index
#asset_count = market_train_df['assetCode'].count()
    # Create a trace
trace1 = go.Bar(
        x = asset_time.index,
        y = asset_time['assetCode'],
        marker=dict(
        color= 'rgb(200,100,5)',
        line=dict(
            color='rgb(200,100,5)',
            width=1.5,
        ), opacity = 0.8
    )
)

layout = go.Layout(
                  title = "Closing prices of {}".format(i),
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )

data = [trace1]
fig=go.Figure(data=data,layout=layout)
py.iplot(fig, filename='basic-line')
fcol_market
type(target)
X = market_train[fcol_market]
target = target
#target
from sklearn.preprocessing import StandardScaler
#df_sc = StandardScaler().fit_transform(X)
#import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(X)
df_sc = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.25, random_state=99)
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
model = Sequential()
model.add(Dense(units=20, activation='relu', input_dim=11))
model.add(Dense(units=1, activation='softmax'))
model.compile(loss='mse',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)
y_pred = model.predict(X_test)
r2_score(y_test,y_pred)
from xgboost import XGBRegressor
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, target, test_size=0.25, random_state=99)

xgb_up = XGBRegressor(n_jobs=-1,n_estimators=200,max_depth=8,eta=0.1)
print('Fitting Up')
xgb_up.fit(X_train,y_train)
print("Accuracy Score: ",accuracy_score(xgb_up.predict(X_test),y_test))
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
x_imputed = my_imputer.fit_transform(X)
from sklearn.preprocessing import StandardScaler
x_imputed_sc = StandardScaler().fit_transform(x_imputed)
x_imputed_sc_df = pd.DataFrame(x_imputed_sc)
x_imputed_sc_df.head()
import time
X_train, X_test, y_train, y_test = model_selection.train_test_split(x_imputed_sc_df, target, test_size=0.25, random_state=99)
start = time.time()
xgb_up = XGBRegressor(n_jobs=-1,n_estimators=20,max_depth=8,eta=0.1)
print('Fitting Up')
xgb_up.fit(X_train,y_train)
end = time.time()
print(end-start)

A=xgb_up.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test,A)