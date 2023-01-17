import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.models import Range1d
from bokeh.models import DatetimeTickFormatter

warnings.filterwarnings("ignore")
btc = pd.read_csv('../input/bitcoinP.csv')
btc.describe()
btc.head()
EMA = pd.DataFrame(index=['Date']) #create new dataframe for EMA
EMA = btc[['Date','Close']][0:].copy() #copy Date and Close Price from btc dataframe
EMA['EMA12'] = EMA['Close'].ewm(12).mean() #12 period moving average
EMA['EMA26'] = EMA['Close'].ewm(26).mean() #26 period moving average
EMA.head()
rows = len(EMA)

#separate negative and possitve EMA signal
for i in range (0,rows):
    if EMA.loc[i,'EMA12'] < EMA.loc[i,'EMA26']:
        EMA.loc[i,'NEGATIVE'] = EMA.loc[i,'EMA12']
    elif EMA.loc[i,'EMA12'] > EMA.loc[i,'EMA26']: 
        EMA.loc[i,'POSSITIVE'] = EMA.loc[i,'EMA12']

fig1 = figure(plot_width = 700, plot_height = 500, 
           x_axis_label = 'Date', y_axis_label = 'Price')

fig1.title.text = "BITCOIN Close Price"
fig1.title.align = "center"

fig1.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d/%m/%Y"],
        days=["%d/%m/%Y"],
        months=["%d/%m/%Y"],
        years=["%d/%m/%Y"],
    )

x = [dt.datetime.strptime(date,'%d-%b-%y').date() for date in EMA['Date']] 
y = btc['Close']
ema26 = EMA['EMA26']
possitive = EMA['POSSITIVE']
negative = EMA['NEGATIVE']

fig1.line(x, y, color = 'darkslategray', alpha=1.0,line_width=1.5, legend='Close Price' )
fig1.line(x, ema26, color = 'darkkhaki',alpha=1.0,line_width=1.5, line_dash=[1, 1],legend='EMA26')
fig1.line(x, possitive, color = 'mediumaquamarine',alpha=1.0,line_width=1.5, legend='EMA12 > EMA26')
fig1.line(x, negative, color = 'indianred',alpha=1.0,line_width=1.5, legend='EMA12 < EMA26')

fig1.legend.location = "top_left"
fig1.legend.border_line_color = "black"

fig1.border_fill_color = "whitesmoke"

output_notebook()

show(fig1)


MACD = pd.DataFrame(index=['Date']) #create new dataframe for MACD
MACD = btc[['Date','Close']][0:].copy() #copy Date and Close Price from btc dataframe
MACD['MACD']=EMA['EMA12']-EMA['EMA26']#calculate MACD
MACD['SIGNAL']= MACD['MACD'].ewm(9).mean()#calculate signal
MACD['HIST']=MACD['MACD']-MACD['SIGNAL']#calculate histogram

HIST = btc[['Date']][0:].copy() 

for i in range(0,rows):
    if MACD.loc[i,'HIST']>0:
        HIST.loc[i,'POSSITIVE']=MACD.loc[i,'HIST']
    else:
        HIST.loc[i,'NEGATIVE']=MACD.loc[i,'HIST']
        

#Create MACD chart

fig2 = figure(plot_width = 700, plot_height = 500, 
           x_axis_label = 'Date', y_axis_label = '')

fig2.title.text = "MACD (12,26,9)"
fig2.title.align = "center"

fig2.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d/%m/%Y"],
        days=["%d/%m/%Y"],
        months=["%d/%m/%Y"],
        years=["%d/%m/%Y"],
    )

x = [dt.datetime.strptime(date,'%d-%b-%y').date() for date in EMA['Date']] 
SIGNAL = MACD['SIGNAL']
MACD = MACD['MACD']
POSSITIVE = HIST['POSSITIVE']
NEGATIVE = HIST['NEGATIVE']

fig2.vbar(x, top = POSSITIVE, width=0.1, color = 'mediumaquamarine')
fig2.vbar(x, top = NEGATIVE, width=0.1,color = 'indianred')
fig2.line(x, SIGNAL, color = 'darkslategray', alpha=1.0,line_width=2, legend='Signal' )
fig2.line(x, MACD, color = 'slategray',alpha=0.75,line_width=1.5, legend='MACD')

fig2.y_range = Range1d(-400, 400) 

fig2.legend.location = "bottom_left"
fig2.legend.border_line_color = "black"

fig2.border_fill_color = "whitesmoke"

output_notebook()

show(fig2)
