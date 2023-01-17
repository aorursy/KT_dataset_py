# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from bokeh.plotting import figure, show, output_file



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input/b3-stock-quotes"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load only the necessary fields

fields = ['DATPRE', 'CODNEG', 'PREABE', 'PREMIN', 'PREULT', 'PREMAX']

df = pd.read_csv(

    #'../input/COTAHIST_A2009_to_A2018P.csv', 

    #'../input/b3-stock-quotes/COTAHIST_A2009_to_A2019.csv', 

    '../input/COTAHIST_A2009_to_A2020_P.csv', 

    usecols=fields

)
# Create a new filtered dataframe

df_filter = df[df.CODNEG == 'PETR4']

df_filter = df_filter[(df_filter['DATPRE'] >= '2020-03-01') & (df_filter['DATPRE'] < '2020-03-31')]

df_filter.head()
from matplotlib.finance import candlestick2_ohlc

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import datetime as datetime
fig, ax = plt.subplots()

candlestick2_ohlc(

    ax,

    df_filter.PREABE, # Opening quote

    df_filter.PREMAX, # High quote

    df_filter.PREMIN, # Low quote

    df_filter.PREULT, # Close quote

    width=0.8)



xdate = [i for i in df_filter.DATPRE]

ax.xaxis.set_major_locator(ticker.MaxNLocator(6))



def mydate(x,pos):

    try:

        return xdate[int(x)]

    except IndexError:

        return ''



ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))



fig.autofmt_xdate()

fig.tight_layout()



plt.show()