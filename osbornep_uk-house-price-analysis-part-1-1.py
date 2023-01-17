import pandas as pd
import numpy as np

import plotly.plotly as py
import plotly.graph_objs as go


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from numpy import arange,array,ones
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from datetime import datetime

import math
dataimport = pd.read_csv('../input/price_paid_records.csv', dtype=object)

(dataimport).head()
len(dataimport)
len(dataimport.drop_duplicates())
# Do not run this in Kaggle, not enough resources.
#hist_trace = go.Histogram( x = dataimport['Price'],
#                          name = 'All House Prices',
#                          xbins = dict(
#                          start = 0,
#                          end = 1000000,
#                          size = 100000
#                          ),
#                          marker = dict(
#                          color = '#EB89B5')
#                         )
#histlayout = go.Layout(
#    title='Distribution of All House Prices',
#    xaxis=dict(
#        title='House Price (£)'
#    ),
#    yaxis=dict(
#        title='Count'
#    ),
#    bargap=0.05,
#    bargroupgap=0.1
#)
#histdata = [hist_trace]
#histfig = go.Figure(data=histdata,layout=histlayout)

# In line plot
#iplot(histfig)

# Shows as new file instead
#plot(histfig)
#Alternative Matplotlib histogram that can't be run in Kaggle due to size of data

#plt.hist(dataimport['Price'])
#plt.title('All House Prices Distribution')
#plt.xlabel('House Price (£)')
#plt.ylabel('Count')

#plt.show()
dataimport['Price'] = dataimport['Price'].astype(str).astype(int)
dataimport3 = dataimport.loc[(dataimport['Price'] < (10000000)) & (dataimport['Price'] > (10000)),]
len(dataimport3)
dataimport3['Date of Transfer'] = pd.to_datetime(dataimport3['Date of Transfer'])

dataimport3['Year'] = dataimport3['Date of Transfer'].dt.year
# Create a list of a day from each month from January 1995 to December 2017
daterange = pd.date_range('1995-01-01','2017-06-30' , freq='1M')
daterange = daterange.union([daterange[-1] ])
daterange = [d.strftime('%d-%m-%Y') for d in daterange]

# Use group by to calculate the number of sales of each month
fulldatafortimeplot = dataimport3.groupby([(dataimport3["Date of Transfer"].dt.year),(dataimport3["Date of Transfer"].dt.month)]).count()
fulldatafortimeplot2 = pd.DataFrame(fulldatafortimeplot['Transaction unique identifier'])
fulldatafortimeplot2['Dates'] = daterange
fulldatafortimeplot2['Dates'] = pd.to_datetime(fulldatafortimeplot2['Dates'], format = '%d-%m-%Y')
fulldatafortimeplot2.columns = ['Count', 'Dates']
fulldatafortimeplot2 = fulldatafortimeplot2
# Plot.ly Timeline plot with range slider
trace_time = go.Scatter(
    x=fulldatafortimeplot2['Dates'],
    y=fulldatafortimeplot2['Count'],
    name = "Number of House Sales",
    line = dict(color = '#7F7F7F'),
    opacity = 0.8)

data_timline = [trace_time]

layout_timeline = go.Layout(
    dict(
    title='Timeline of the Number of House Sales in the UK between 1995 and 2017',
    xaxis=dict(

        rangeslider=dict(),
        type='date'
    ),
    annotations = [
        dict(
        x = datetime.strptime('23-06-2016', '%d-%m-%Y'),
        y = 84927,
        xref = 'x',
        yref = 'y',
        text = 'UK Referendum',
        showarrow = True,
        arrowhead = 7,
        ax = 0,
        ay = -40
        ),
        dict(
        x = datetime.strptime('01-12-2007', '%d-%m-%Y'),
        y = 104283,
        xref = 'x',
        yref = 'y',
        text = 'Financial Crash',
        showarrow = True,
        arrowhead = 7,
        ax = 0,
        ay = -40
        )
    ]

)
)

fig = dict(data=data_timline, layout=layout_timeline)
iplot(fig)