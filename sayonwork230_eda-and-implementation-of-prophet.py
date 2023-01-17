import pandas as pd

import numpy as np

import plotly.offline as plo

import plotly.graph_objs as go

import matplotlib.pyplot as plt

from fbprophet import Prophet
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_excel('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.xlsx', index_col='Time Serie')

df.index = pd.to_datetime(df.index, format="%Y-%m-%d")

#df.drop(columns={'Unnamed: 0'}, axis=1, inplace=True)

df.columns = [i.split('-')[1].strip().split('/')[0].lower() for i in df.columns]

df.head()
temp = df.iloc[[-1],:]

data = [go.Bar(

    x = temp[colname],

    y = temp.index,

    name = colname,

    orientation = 'h'

)for colname in temp.columns]



layout = go.Layout(

    title = "Exchange rates",

    template = 'plotly_dark'

)



fig = go.Figure(data=data,layout=layout)

plo.iplot(fig)
def curr_graph(currency):

    temp = df.loc[:,[currency]]

    data = go.Scatter(

        x = temp.index,

        y = temp[currency],

        mode = 'lines',

        marker = dict(color='blue'),

        name = currency

    )



    layout = go.Layout(

        title = '{0}/us$'.format(currency),

        template = 'plotly_dark'

    )

    fig = go.Figure(data=data,layout=layout)

    plo.iplot(fig)
# Selecting those currency whose exchange rate is > 71 in the present day

select_currency = ['sri lankan rupee', 'yen', 'won', 'indian rupee']

for currency in select_currency:

    curr_graph(currency)
select_currency = ['sri lankan rupee', 'yen', 'indian rupee']

temp = df.loc[:,select_currency]

data = [go.Scatter(

    x = temp.index,

    y = temp[currency],

    mode = 'lines',

    name = currency

)for currency in select_currency]



layout = go.Layout(

    title = '{0}/us$'.format(currency),

    template = 'plotly_dark'

)

fig = go.Figure(data=data,layout=layout)

plo.iplot(fig)
data = df.loc['2015-01-02':,['indian rupee']]

data = data.replace('ND',np.nan)

data = data.bfill().reset_index()

data = data.rename(columns={

    'Time Serie' : 'ds',

    'indian rupee' : 'y'

})

data
training_data = data[data['ds']<='2018-12-31']

testing_data = data[data['ds']>='2019-01-01']
testing_data
prophet = Prophet()

prophet.fit(training_data)
future_dates = prophet.make_future_dataframe(periods=365,freq='D', include_history=False)
prediction = prophet.predict(future_dates)
pre_data = df.loc['2015-01-02':,['indian rupee']].replace('ND',np.nan).bfill()

pre_data['indian rupee'] = pre_data['indian rupee'].apply(lambda x: float(x))

fig,axes = plt.subplots(figsize=(20,8))

plt.plot(pre_data.index,pre_data.values,axes=axes,color='red')

prophet.plot(prediction, ax=axes)

plt.show()
temp = prediction[['ds','yhat']].rename(columns={

    'yhat' : 'y'

})
MAPE = (abs(testing_data.set_index('ds')-temp.set_index('ds'))/testing_data.set_index('ds')).mean() *100

print("MAPE =",round(float(MAPE.values),3))