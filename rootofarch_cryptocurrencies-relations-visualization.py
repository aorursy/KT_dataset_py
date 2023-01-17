import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly

from plotly.offline import iplot

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from IPython.display import clear_output

import random

import time

import os

print(os.listdir("../input"))
csv_list = os.listdir("../input")

data_list = [x[:-4] for x in csv_list]

data_str_list = data_list.copy()

for i in range(0,len(csv_list)):

    temp = "../input/" + csv_list[i]

    data_list[i] = pd.read_csv(temp)
def printer(string):

    for i in range(0,len(data_list)):

        if data_str_list[i].count(string) > 0:

            print(data_str_list[i],"\n",data_list[i].columns,"\n")



printer("dataset")

printer("price")
price_datas=[]

for i in range(0,len(data_list)):

    if data_str_list[i].count("price") > 0:

        data_list[i]["Name"] = data_str_list[i][:-6]

        data_list[i].columns = [col.replace(' ', '_') for col in data_list[i].columns]

        #data_list[i].Date = pd.to_datetime(price_datas[i].Date, infer_datetime_format=True)

        price_datas.append(data_list[i])



price_datas[0].columns
print("type of dates:\t",type(price_datas[4].Date[0]))

price_datas[4].head()
for i in range(0,len(price_datas)):

    price_datas[i].Date = pd.to_datetime(price_datas[i].Date, infer_datetime_format=True)
print("type of dates:\t",type(price_datas[4].Date[0]))

price_datas[4].head()
price_comp_list = pd.concat(price_datas,ignore_index=True)

price_comp_list.head()
price_comp_list.info()
#price_comp_list.Market_Cap.str.replace(",","").astype(int)
ClearData = price_comp_list.copy()

ClearData = ClearData[ClearData.Market_Cap != "-"]

ClearData.Market_Cap = ClearData.Market_Cap.str.replace(",","").astype(int)

ClearData = ClearData[ClearData.Volume != "-"]

ClearData.Volume = ClearData.Volume.str.replace(",","").astype(int)

ClearData.head()
ClearData.info()
pd.options.display.float_format = '{:,.2f}'.format

MeanValues = ClearData.groupby("Name").mean()

MeanValues
def PlotPieChart(Name,label,value):

    trace = go.Pie(labels=label, values=value)

    

    data = [trace]

    layout = dict(title = str(Name))

    fig = dict(data=data, layout=layout)

    iplot(fig)
PlotPieChart("Cryptocurrency MarketCaps",MeanValues.index,MeanValues.Market_Cap)

PlotPieChart("Cryptocurrency Volumes",MeanValues.index,MeanValues.Volume)
def traceGraph(CryptocurrencyName):

    ClearCrypto = ClearData[ClearData.Name == CryptocurrencyName].set_index("Date")

    

    trace_high = go.Scatter(x=list(ClearCrypto.index),

                            y=list(ClearCrypto.High),

                            name='High',

                            line=dict(color='#33CFA5'))



    trace_high_avg = go.Scatter(x=list(ClearCrypto.index),

                                y=[ClearCrypto.High.mean()]*len(ClearCrypto.index),

                                name='High Average',

                                visible=False,

                                line=dict(color='#33CFA5', dash='dash'))



    trace_low = go.Scatter(x=list(ClearCrypto.index),

                           y=list(ClearCrypto.Low),

                           name='Low',

                           line=dict(color='#F06A6A'))



    trace_low_avg = go.Scatter(x=list(ClearCrypto.index),

                               y=[ClearCrypto.Low.mean()]*len(ClearCrypto.index),

                               name='Low Average',

                               visible=False,

                               line=dict(color='#F06A6A', dash='dot'))



    data = [trace_high, trace_high_avg, trace_low, trace_low_avg]



    high_annotations=[dict(y=ClearCrypto.High.mean(),

                           text='High Average:<br>'+str(ClearCrypto.High.mean()),

                           ax=0, ay=-50),

                      dict(x=ClearCrypto.High.idxmax(),

                           y=ClearCrypto.High.max(),

                           text='High Max:<br>'+str(ClearCrypto.High.max()),

                           ax=0, ay=-50)]

    low_annotations=[dict(y=ClearCrypto.Low.mean(),

                          text='Low Average:<br>'+str(ClearCrypto.Low.mean()),

                          ax=0, ay=50),

                     dict(x=ClearCrypto.High.idxmin(),

                          y=ClearCrypto.Low.min(),

                          text='Low Min:<br>'+str(ClearCrypto.Low.min()),

                          ax=0, ay=50)]



    updatemenus = list([

        dict(type="buttons",

             active=-1,

             buttons=list([

                dict(label = 'High',

                     method = 'update',

                     args = [{'visible': [True, True, False, False]},

                             {'title': CryptocurrencyName.capitalize() + ' High',

                              'annotations': high_annotations}]),

                dict(label = 'Low',

                     method = 'update',

                     args = [{'visible': [False, False, True, True]},

                             {'title': CryptocurrencyName.capitalize() + ' Low',

                              'annotations': low_annotations}]),

                dict(label = 'Both',

                     method = 'update',

                     args = [{'visible': [True, True, True, True]},

                             {'title': CryptocurrencyName.capitalize() + ' Both',

                              'annotations': high_annotations+low_annotations}]),

                dict(label = 'Reset',

                     method = 'update',

                     args = [{'visible': [True, False, True, False]},

                             {'title': CryptocurrencyName.capitalize(),

                              'annotations': []}])

            ]),

        )

    ])



    layout = dict(title=CryptocurrencyName.capitalize(), showlegend=False,

                  updatemenus=updatemenus)



    fig = dict(data=data, layout=layout)

    iplot(fig)
traceGraph("bitcoin")
#while True:

#    traceGraph(MeanValues.index[random.randint(0,len(MeanValues.index))])

#    time.sleep(10)

#    clear_output(wait=True)
def MarketCapGraph(currencyList):

    gf = ClearData.groupby('Name')

    data = []



    for currency in currencyList[::-1]:

        group = gf.get_group(currency)

        dates = group['Date'].tolist()

        date_count = len(dates)

        marketCap = group['Market_Cap'].tolist()

        zeros = [0] * date_count



        data.append(dict(

            type='scatter3d',

            mode='lines',

            x=dates + dates[::-1] + [dates[0]],  # year loop: in incr. order then in decr. order then years[0]

            y=[currency] * date_count,

            z=marketCap + zeros + [marketCap[0]],

            name=currency,

            line=dict(

                width=4

            ),

        ))



    layout = dict(

        title='Cryptocurrencies Market Capitalizations',

        scene=dict(

            xaxis=dict(title='Dates'),

            yaxis=dict(title='Cryptocurrencies'),

            zaxis=dict(title='Market Capitalizations'),

            camera=dict(

                eye=dict(x=-1.7, y=-1.7, z=0.5)

            )

        )

    )



    fig = dict(data=data, layout=layout)

    iplot(fig)
MarketCapGraph(MeanValues.sort_values(by=['Market_Cap'],ascending=False).head(5).index)
MarketCapGraph(MeanValues.index)