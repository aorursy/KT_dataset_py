import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import warnings

warnings.filterwarnings('ignore')

import os

import pandas as pd



from plotly import __version__

print(__version__) # need 1.9.0 or greater

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import graph_objs as go

import requests

import pandas as pd



init_notebook_mode(connected = True)



def plotly_df(df, title = ''):

    data = []

    

    for column in df.columns:

        trace = go.Scatter(

            x = df.index,

            y = df[column],

            mode = 'lines',

            name = column

        )

        data.append(trace)

    

    layout = dict(title = title)

    fig = dict(data = data, layout = layout)

    iplot(fig, show_link=False)

    

%matplotlib inline

import matplotlib.pyplot as plt

from scipy import stats

import statsmodels.api as sm
habr_df = pd.read_csv('/kaggle/input/howpop/howpop_train.csv')
habr_df.head()
habr_df['published'] = pd.to_datetime(habr_df.published)

habr_df = habr_df[['published', 'url']]

habr_df = habr_df.drop_duplicates()
habr_df.head()
aggr_habr_df = habr_df.groupby('published')[['url']].count()

aggr_habr_df.columns = ['posts']
aggr_habr_df.head()
aggr_habr_df.posts.value_counts()
aggr_habr_df = aggr_habr_df.resample('D').apply(sum)

plotly_df(aggr_habr_df.resample('W').apply(sum), 

          title = 'Опубликованные посты на Хабрахабре')
# !pip install pystan

# !pip install fbprophet

from fbprophet import Prophet
predictions = 30



df = aggr_habr_df.reset_index()

df.columns = ['ds', 'y']

df.tail()
train_df = df[:-predictions]
m = Prophet()

m.fit(train_df)
future = m.make_future_dataframe(periods=30)

future.tail()
forecast = m.predict(future)

forecast.tail()
print(', '.join(forecast.columns))
m.plot(forecast)
m.plot_components(forecast)
cmp_df = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(df.set_index('ds'))
cmp_df.head()
import numpy as np

cmp_df['e'] = cmp_df['y'] - cmp_df['yhat']

cmp_df['p'] = 100*cmp_df['e']/cmp_df['y']

np.mean(abs(cmp_df[-predictions:]['p'])), np.mean(abs(cmp_df[-predictions:]['e']))
def invboxcox(y, lmbda):

    if lmbda == 0:

        return(np.exp(y))

    else:

        return(np.exp(np.log(lmbda * y + 1) / lmbda))
train_df2 = train_df.copy().fillna(14)

train_df2 = train_df2.set_index('ds')

train_df2.y = train_df2.y.apply(lambda x: 1 if x==0 else x)

train_df2['y'], lmbda_prophet = stats.boxcox(train_df2['y'])
train_df2.reset_index(inplace=True)



m2 = Prophet()

m2.fit(train_df2)

future2 = m2.make_future_dataframe(periods=30)



forecast2 = m2.predict(future2)

forecast2['yhat'] = invboxcox(forecast2.yhat, lmbda_prophet)

forecast2['yhat_lower'] = invboxcox(forecast2.yhat_lower, lmbda_prophet)

forecast2['yhat_upper'] = invboxcox(forecast2.yhat_upper, lmbda_prophet)



cmp_df2 = forecast2.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(df.set_index('ds'))



cmp_df2['e'] = cmp_df2['y'] - cmp_df2['yhat']

cmp_df2['p'] = 100*cmp_df2['e']/cmp_df2['y']

np.mean(abs(cmp_df2[-predictions:]['p'])), np.mean(abs(cmp_df2[-predictions:]['e']))
def show_forecast(cmp_df, num_predictions, num_values):

    upper_bound = go.Scatter(

        name='Upper Bound',

        x=cmp_df.tail(num_predictions).index,

        y=cmp_df.tail(num_predictions).yhat_upper,

        marker=dict(color="green"),)





    forecast = go.Scatter(

        name='Prediction',

        x=cmp_df.tail(predictions).index,

        y=cmp_df.tail(predictions).yhat,

        mode='lines',

        line=dict(color='blue'),

    )



    lower_bound = go.Scatter(

        name='Lower Bound',

        x=cmp_df.tail(num_predictions).index,

        y=cmp_df.tail(num_predictions).yhat_lower,

        marker=dict(color="yellow"),)



    fact = go.Scatter(

        name='Fact',

        x=cmp_df.tail(num_values).index,

        y=cmp_df.tail(num_values).y,

        marker=dict(color="red"),

        mode='lines',

    )



 

    data = [lower_bound, upper_bound, forecast, fact]



    layout = go.Layout(

        yaxis=dict(title='Посты'),

        title='Опубликованные посты на Хабрахабре',

        showlegend = False)



    fig = go.Figure(data=data, layout=layout)

    iplot(fig, show_link=False)
show_forecast(cmp_df, predictions, 200)