from IPython.display import HTML



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

<form action="javascript:code_toggle()"><input type="submit" value="Toggle on/off the raw code."></form>''')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import matplotlib.pyplot as plt

%matplotlib inline



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go



import seaborn as sns

sns.set(style="white", palette="muted", color_codes=True)



init_notebook_mode()
df = pd.read_csv("../input/istanbul_stock_exchange.csv", low_memory = False)

df.head(5)
def chk_missing(df):

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum()/len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    return mis_val_table_ren_columns 



chk_missing(df)
df['date'] = pd.to_datetime(df['date'])

df.head(3)
print("The minimum given date", min(df.date), ". The maximum given date", max(df.date))
trace_tl = go.Scatter(

    x=df.date,

    y=df['TL BASED ISE'],

    name = "TL BASED ISE",

    line = dict(color = '#17BECF'),

    opacity = 0.8)



trace_usd = go.Scatter(

    x=df.date,

    y=df['USD BASED ISE'],

    name = "USD BASED ISE",

    line = dict(color = '#7F7F7F'),

    opacity = 0.8)



data = [trace_tl, trace_usd]



layout = dict(

    title='ISE Stock',

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=6,

                     label='6m',

                     step='month',

                     stepmode='backward'),

                dict(count=12,

                    label='12m',

                    step='month',

                    stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(),

        type='date'

    )

)



fig = dict(data=data, layout=layout)



iplot(fig, filename='time-series-ise')  
def plotly_1plot (df, col_date_nm, col_idx_nm):

    trace = go.Scatter(

    x=df[col_date_nm],

    y=df[col_idx_nm],

    name = col_idx_nm,

    line = dict(color = '#17BECF'),

    opacity = 0.8)

    

    data = [trace]

    

    layout = dict(

        title= col_idx_nm,

        xaxis=dict(

            rangeselector=dict(

                buttons=list([

                    dict(count=1,

                         label='1m',

                         step='month',

                         stepmode='backward'),

                    dict(count=6,

                         label='6m',

                         step='month',

                         stepmode='backward'),

                    dict(count=12,

                        label='12m',

                        step='month',

                        stepmode='backward'),

                    dict(step='all')

                ])

            ),

            rangeslider=dict(),

            type='date'

        )

    )

    

    fig = dict(data=data, layout=layout)

    return(fig)



iplot(plotly_1plot(df, 'date', 'SP'), filename='time-series-SP') 

iplot(plotly_1plot(df, 'date', 'DAX'), filename='time-series-DAX') 
iplot(plotly_1plot(df, 'date', 'FTSE'), filename='time-series-FTSE') 
iplot(plotly_1plot(df, 'date', 'NIKKEI'), filename='time-series-NIKKEI') 
iplot(plotly_1plot(df, 'date', 'BOVESPA'), filename='time-series-BOVESPA') 
trace_eu = go.Scatter(

    x=df.date,

    y=df['EU'],

    name = "Europe",

    line = dict(color = '#17BECF'),

    opacity = 0.8)



trace_em = go.Scatter(

    x=df.date,

    y=df['EM'],

    name = "Emerging Market",

    line = dict(color = 'red'),

    opacity = 0.8)



data = [trace_eu, trace_em]



layout = dict(

    title='MSCI EU vs. EM',

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=6,

                     label='6m',

                     step='month',

                     stepmode='backward'),

                dict(count=12,

                    label='12m',

                    step='month',

                    stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(),

        type='date'

    )

)



fig = dict(data=data, layout=layout)



iplot(fig, filename='time-series-msci-eu-em')  