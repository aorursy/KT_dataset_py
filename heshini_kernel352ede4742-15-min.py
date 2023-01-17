# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# split_date1 = '2019-09-05T19:25:00.000Z'

# split_date2='2019-09-29T23:54:00.000Z'

# train = df.loc[df['timestamp'] <= split_date1].copy()

# test = df.loc[(df['timestamp']> split_date1)&(df['timestamp'] <= split_date2)].copy()

# validation=df.loc[df['timestamp'] >split_date2].copy()
import pandas as pd

j_test = pd.read_csv("../input/15-min-data/june15.csv")

split_date1 = '2019-06-12T19:25:00.000Z'

split_date2='2019-06-19T23:54:00.000Z'

train = j_test.loc[j_test['timestamp'] <= split_date1].copy()

test = j_test.loc[(j_test['timestamp']> split_date1)&(j_test['timestamp'] <= split_date2)].copy()

validation=j_test.loc[j_test['timestamp'] >split_date2].copy()

import plotly.express as px

fig = px.line(test, x='timestamp', y='mb')

fig.show()
import pandas as pd

aug15 = pd.read_csv("../input/15-min-data/aug15.csv")

july15 = pd.read_csv("../input/15-min-data/july15.csv")

june15 = pd.read_csv("../input/15-min-data/june15.csv")

oct15 = pd.read_csv("../input/15-min-data/oct15.csv")

sep15 = pd.read_csv("../input/15-min-data/sep15.csv")
import plotly.express as px

fig = px.line(june15, x='timestamp', y='mb')

fig.show()
# from pylab import rcParams

# import statsmodels.api as sm

# import matplotlib

# rcParams['figure.figsize'] = 18, 8

# decomposition = sm.tsa.seasonal_decompose(june15.mb, model='additive')

# fig = decomposition.plot()

# plt.show()
import plotly.express as px

fig = px.line(july15, x='timestamp', y='mb')

fig.show()
import plotly.express as px

fig = px.line(aug15, x='timestamp', y='mb')

fig.show()
import plotly.express as px

fig = px.line(sep15, x='timestamp', y='mb')

fig.show()
import plotly.express as px

fig = px.line(oct15, x='timestamp', y='mb')

fig.show()
# def create_features(df, label=None):

#     """

#     Creates time series features from datetime index

#     """

#     df['time']=df['timestamp']

#     df['date'] = df['timestamp']

#     df['day']=df['date'].dt.day

#     df['hour'] = df['date'].dt.hour

#     df['dayofweek'] = df['date'].dt.dayofweek

#     df['month'] = df['date'].dt.month

#     #df['weekofthemonth']=df['date'].dt.weekofmonth

#     df['dayofmonth'] = df['date'].dt.day

#     df['weekofyear'] = df['date'].dt.weekofyear

#     df['min'] = df['date'].dt.minute

#     df['mb']=df['mb']

#     X = df[['timestamp','day','hour','dayofweek','month','dayofmonth','weekofyear','min','mb']]

#     return X
# import plotly.plotly as py

# import plotly.graph_objs as go



# import pandas as pd



# trace_high = go.Scatter(

#                 x=mon_june[],

#                 y=df['AAPL.High'],

#                 name = "AAPL High",

#                 line = dict(color = '#17BECF'),

#                 opacity = 0.8)



# trace_low = go.Scatter(

#                 x=df.Date,

#                 y=df['AAPL.Low'],

#                 name = "AAPL Low",

#                 line = dict(color = '#7F7F7F'),

#                 opacity = 0.8)



# data = [trace_high,trace_low]



# layout = dict(

#     title = "Manually Set Date Range",

#     xaxis = dict(

#         range = ['2016-07-01','2016-12-31'])

# )



# fig = dict(data=data, layout=layout)

# py.iplot(fig, filename = "Manually Set Range")
# import matplotlib.pyplot as plt



# fig, (ax1, ax2) = plt.subplots(2)

# fig.suptitle('each mon of june')

# ax1.plot(mon_june[mon_june.day==2])

# ax2.plot(mon_june[mon_june.day==9])
# import plotly.graph_objects as go

# from plotly.subplots import make_subplots



# fig = make_subplots(rows=2, cols=2,

#                     specs=[[{"secondary_y": True}, {"secondary_y": True}],

#                            [{"secondary_y": True}, {"secondary_y": True}]])




# # Top left

# fig.add_trace(

#     go.Scatter(x=, y=mJ2.mb, name="yaxis data"),

#     row=1, col=1, secondary_y=False)





# # Top right

# fig.add_trace(

#     go.Scatter(x=[1, 2, 3], y=[2, 52, 62], name="yaxis3 data"),

#     row=1, col=2, secondary_y=False,

# )







# # Bottom left

# fig.add_trace(

#     go.Scatter(x=[1, 2, 3], y=[2, 52, 62], name="yaxis5 data"),

#     row=2, col=1, secondary_y=False,

# )





# # Bottom right

# fig.add_trace(

#     go.Scatter(x=[1, 2, 3], y=[2, 52, 62], name="yaxis7 data"),

#     row=2, col=2, secondary_y=False,

# )





# fig.show()


