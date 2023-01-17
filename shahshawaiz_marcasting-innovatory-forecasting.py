# install libs

# !pip install fbprophet
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fbprophet import Prophet



# read dataset

train = pd.read_csv("../input/dataset/train.csv")

test = pd.read_csv("../input/dataset/test.csv")



train.head()
# create df for prophet

def create_df(ds, y):    

    df = pd.DataFrame(columns=['ds','y'])

    df['ds'] = ds

    df = df.set_index('ds')

    df['y'] = y.values

    print(y)

    df.reset_index(drop=False,inplace=True)

    return df





train_store_1 = train[train.store==1]

df = create_df(train_store_1.date, train_store_1.sales)

df.tail()
# fit model

m = Prophet()

m.fit(df)
# future df next n days

def create_df_future(m, days):

    return m.make_future_dataframe(periods=days)    



def predict(m, forecast):

    forecast = m.predict(forecast)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()



future_df = create_df_future(m, 10)

forecast = predict(m, future_df)
def visualize_forecat(m, forecast):

    fig = m.plot(forecast)

    fig

    

    return fig



def visualize_forecat_componenets(m, forecast):

    fig = m.plot_components(forecast)

    fig

    

    return fig
visualize_forecat(m, forecast)
visualize_forecat_componenets(m, forecast)