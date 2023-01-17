import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

from plotly import tools

import plotly as py

import plotly.graph_objs as go

py.offline.init_notebook_mode(connected = True)

import os



apple_stocks = pd.read_csv(r'../input/AAPL.csv')

ibm_stocks = pd.read_csv(r'../input/IBM.csv')

cococola_stocks = pd.read_csv(r'../input/KO.csv')

nike_stocks = pd.read_csv(r'../input/NKE.csv')



tmp = pd.DataFrame()

dow_jones = pd.DataFrame()

for feature in ['open' , 'high' , 'low' , 'close']:

    for csv in os.listdir('../input'):

        tmp[csv] = pd.read_csv(r'../input/'+csv)[feature]

    dow_jones[feature] = np.mean(tmp , axis = 1)

    tmp = pd.DataFrame()

    

dow_jones['date'] = apple_stocks['date']



plt.style.use('fivethirtyeight')

def special_plot(dates , open_ ,close_ , high_ , low_

                 ,change_pos , change_neg  ,title , title_1 , title_2 , title_3):

    

    trace1 = go.Scatter(

        x = dates,

        y = close_,

        fill = 'tonexty',

        mode = 'lines',

        name = 'lines'

    )

    trace2 = go.Ohlc(

        x = dates,

        open = open_,

        high = high_,

        low = low_,

        close = close_,

        yaxis = 'y2'

        )

    trace3 = go.Scatter(

        x = dates,

        y = change_pos,

        mode =  'lines',

        fill= 'tonexty',

        yaxis = 'y3',

        line = dict(

            color = 'rgb(121, 239, 67)'

        )

        )

    trace4 = go.Scatter(

        x = dates,

        y = change_neg,

        mode =  'lines',

        fill= 'tozeroy',

        yaxis = 'y3',

        line=dict(

            color='rgb(237, 52, 73)',

        )

        )

    fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],

                              subplot_titles=(title_1,title_2,title_3))



    fig.append_trace(trace1, 1, 1)

    fig.append_trace(trace3, 1, 2)

    fig.append_trace(trace4, 1, 2)

    fig.append_trace(trace2, 2 ,1)



    fig['layout'].update(showlegend=False, title=title)

    py.offline.iplot(fig)



def interactive_line(x , y , title):

    trace0 = go.Scatter(

        x = x,

        y = y,

        fill = 'tonexty',

        mode = 'lines',

        name = 'lines'

        )

    layout = go.Layout(

        title = title

        )

    data = [trace0]

    fig = go.Figure(data = data , layout = layout)

    py.offline.iplot(fig)

    

def calculas(y,n = 7):

    change_pos = []

    change_neg = []

    for i in range(y.shape[0]):

        if i + n == y.shape[0]:

            break

        c = (y[i + n] -  y[i])/n

        if c >= 0 :

            change_pos.append(c)

            change_neg.append(0)

        else:

            change_neg.append(c)

            change_pos.append(0)

    return change_pos , change_neg



from IPython.display import HTML

HTML('''

<script>

  function code_toggle() {

    if (code_shown){

      $('div.input').hide('500');

      $('#toggleButton').val('Show Code')

    } else {

      $('div.input').show('500');

      $('#toggleButton').val('Hide Code')

    }

    code_shown = !code_shown

  }



  $( document ).ready(function(){

    code_shown=false;

    $('div.input').hide()

  });

</script>

<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
dow_change_pos , dow_change_neg = calculas(dow_jones['close'] , 30)
special_plot(dates = dow_jones['date'] , open_ = dow_jones['open'],

            close_ = dow_jones['close'] , high_ = dow_jones['high'],

            low_ = dow_jones['low'] , change_pos = dow_change_pos , 

            change_neg = dow_change_neg , title = 'Dow Jones Stocks.' , 

             title_1 = 'Stocks', title_2 = 'Monthly Changes in stock ' , 

             title_3 = 'Open-high-low-close')
apple_change_pos , apple_change_neg  = calculas(apple_stocks['close'], 30)
special_plot(dates = apple_stocks['date'] , open_ = apple_stocks['open'],

            close_ = apple_stocks['close'] , high_ = apple_stocks['high'],

            low_ = apple_stocks['low'] , change_pos = apple_change_pos , 

            change_neg = apple_change_neg , title = 'Apple Inc.' , 

             title_1 = 'Stocks', title_2 = 'Monthly Changes in stock ' , 

             title_3 = 'Open-high-low-close')
ibm_change_pos , ibm_change_neg = calculas(ibm_stocks['close'] , 30)
special_plot(dates = ibm_stocks['date'] , open_ = ibm_stocks['open'] , 

            close_ = ibm_stocks['close'] , high_ = ibm_stocks['high'] , 

            low_ = ibm_stocks['low'] , change_pos = ibm_change_pos , 

            change_neg = ibm_change_neg , title = 'IBM Inc.' , 

            title_1 = 'Stocks' , title_2 = 'Monthly changes in stocks' , 

            title_3 = 'Open-high-low-close')
nike_change_pos , nike_change_neg  = calculas(nike_stocks['close'] , 30)
special_plot(dates = nike_stocks['date'] , open_ = nike_stocks['open'] , 

            close_ = nike_stocks['close'] , high_ = nike_stocks['high'] , 

            low_ = nike_stocks['low'] , change_pos = nike_change_pos , 

            change_neg = nike_change_neg , title = 'Nike Inc.' , 

            title_1 = 'Stocks' , title_2 = 'Monthly changes in stocks' , 

            title_3 = 'Open-high-low-close')
cococola_change_pos , cococola_change_neg = calculas(cococola_stocks['close'] , 30)
special_plot(dates = cococola_stocks['date'] , open_ = cococola_stocks['open'] , 

            close_ = cococola_stocks['close'] , high_ = cococola_stocks['high'] , 

            low_ = cococola_stocks['low'] , change_pos = cococola_change_pos , 

            change_neg = cococola_change_neg , title = 'COCO-COLA Inc.' , 

            title_1 = 'Stocks' , title_2 = 'Monthly changes in stocks' , 

            title_3 = 'Open-high-low-close')