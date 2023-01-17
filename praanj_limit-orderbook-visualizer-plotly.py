import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dec_test1 = np.loadtxt('../input/Train_Dst_NoAuction_DecPre_CF_7.txt')
df1 = dec_test1[:40, :].T

df = pd.DataFrame(df1)

df.head()
# Ask already followed natural order

dfAskPrices = df.loc[:, range(0,40,4)]

dfAskVolumes = df.loc[:, range(1,40,4)]



# Bid follows reversed natural order

dfBidPrices = df.loc[:, range(2,40,4)]

dfBidVolumes = df.loc[:, range(3,40,4)]



# Reverse Bid price and volumnes to make them follow natural order

dfBidPrices = dfBidPrices[dfBidPrices.columns[::-1]]

dfBidVolumes = dfBidVolumes[dfBidVolumes.columns[::-1]]



# Concatenate Bid and Ask together to form complete orderbook picture

dfPrices = dfBidPrices.join(dfAskPrices, how='outer')

dfVolumnes = dfBidVolumes.join(dfAskVolumes, how='outer')



#Rename columns starting from 1->20

dfPrices.columns = range(1, 21)

dfVolumnes.columns = range(1, 21)



dfPrices.head()
dfVolumnes.head()
fig = go.Figure()



for i in dfPrices.columns: 

    fig.add_trace(go.Scatter(y=dfPrices.head(40)[i]))



fig.update_layout(

    title='10 price levels of each side of the orderbook, bar size represents volume',

    xaxis_title="Time snapshot index",

    yaxis_title="Price levels",

    template='plotly_dark'

)



fig.show()
px.bar(dfVolumnes.head(5).transpose(), orientation='h')
colors = ['lightslategrey',] * 10

colors = colors + ['crimson',] * 10
fig = go.Figure()

timestamp = 100000



fig.add_trace(go.Bar(

    y= ['price-'+'{:.4f}'.format(x) for x in dfPrices[:timestamp].values[0].tolist()],

    x=dfVolumnes[:timestamp].values[0].tolist(),

    orientation='h',

    marker_color=colors

))



fig.update_layout(

    title='10 price levels of each side of the orderbook, bar size represents volume',

    xaxis_title="Volume",

    yaxis_title="Price levels",

    template='plotly_dark'

)



fig.show()
fig = make_subplots(rows=1, cols=2)



for i in dfPrices.columns: 

    fig.add_trace(go.Scatter(y=dfPrices.head(20)[i]), row=1, col=1)



timestamp = 500000



fig.add_trace(go.Bar(

    y= ['price-'+'{:.4f}'.format(x) for x in dfPrices[:timestamp].values[0].tolist()],

    x=dfVolumnes[:timestamp].values[0].tolist(),

    orientation='h',

    marker_color=colors

), row=1, col=2)



fig.update_layout(

    title='10 price levels of each side of the orderbook for multiple time points, bar size represents volume',

    xaxis_title="Time snapshot",

    yaxis_title="Price levels",

    template='plotly_dark'

)



fig.show()
widthOfTime = 100;



fig = go.Figure(

    data=[go.Scatter(x=dfPrices.index[:widthOfTime].tolist(), y=dfPrices[:widthOfTime][1].tolist(),

                     name="frame",

                     mode="lines",

                     line=dict(width=2, color="blue")),

          ],

    layout=go.Layout(width=1000, height=400,

#                      xaxis=dict(range=[0, 100], autorange=False, zeroline=False),

#                      yaxis=dict(range=[0, 1], autorange=False, zeroline=False),

                     title="10 price levels of each side of the orderbook",

                     xaxis_title="Time snapshot index",

                     yaxis_title="Price levels",

                     template='plotly_dark',

                     hovermode="closest",

                     updatemenus=[dict(type="buttons",

                                       showactive=True,

                                       x=0.01,

                                       xanchor="left",

                                       y=1.15,

                                       yanchor="top",

                                       font={"color":'blue'},

                                       buttons=[dict(label="Play",

                                                     method="animate",

                                                     args=[None])])]),



    frames=[go.Frame(

        data=[go.Scatter(

            x=dfPrices.iloc[k:k+widthOfTime].index.tolist(),

            y=dfPrices.iloc[k:k+widthOfTime][1].tolist(),

            mode="lines",

            line=dict(color="blue", width=2))

        ]) for k in range(widthOfTime, 1000)]

)



fig.show()
timeStampStart = 100



fig = go.Figure(

    data=[go.Bar(y= ['price-'+'{:.4f}'.format(x) for x in dfPrices[:timeStampStart].values[0].tolist()],

                 x=dfVolumnes[:timeStampStart].values[0].tolist(),

                 orientation='h',

                 name="priceBar",

                 marker_color=colors),

          ],

    layout=go.Layout(width=800, height=450,

                     title="Volume of 10 buy, sell price levels of an orderbook",

                     xaxis_title="Volume",

                     yaxis_title="Price levels",

                     template='plotly_dark',

                     hovermode="closest",

                     updatemenus=[dict(type="buttons",

                                       showactive=True,

                                       x=0.01,

                                       xanchor="left",

                                       y=1.15,

                                       yanchor="top",

                                       font={"color":'blue'},

                                       buttons=[dict(label="Play",

                                                     method="animate",

                                                     args=[None])])]),

    frames=[go.Frame(

        data=[go.Bar(y= ['price-'+'{:.4f}'.format(x) for x in dfPrices.iloc[k].values.tolist()],

                     x=dfVolumnes.iloc[k].values.tolist(),

                     orientation='h',

                     marker_color=colors)],

        layout=go.Layout(width=800, height=450,

                     title="Volume of 10 buy, sell price levels of an orderbook [Snapshot=" + str(k) +"]",

                     xaxis_title="Volume",

                     yaxis_title="Price levels",

                     template='plotly_dark',

                     hovermode="closest")) for k in range(timeStampStart, 500)]

)



fig.show()