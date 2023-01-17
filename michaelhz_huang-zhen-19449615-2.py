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
!pip install xlrd

!pip install plotly

!pip install chart_studio
import pandas as pd

import numpy as np





def excel_to_df(excel_sheet):

    df = pd.read_excel(excel_sheet)

    df.dropna(how='all', inplace=True)



    index_PL = int(df.loc[df['Data provided by SimFin']=='Profit & Loss statement'].index[0])

    index_CF = int(df.loc[df['Data provided by SimFin']=='Cash Flow statement'].index[0])

    index_BS = int(df.loc[df['Data provided by SimFin']=='Balance Sheet'].index[0])



    df_PL = df.iloc[index_PL:index_BS-1, 1:]

    df_PL.dropna(how='all', inplace=True)

    df_PL.columns = df_PL.iloc[0]

    df_PL = df_PL[1:]

    df_PL.set_index("in million USD", inplace=True)

    (df_PL.fillna(0, inplace=True))

    



    df_BS = df.iloc[index_BS-1:index_CF-2, 1:]

    df_BS.dropna(how='all', inplace=True)

    df_BS.columns = df_BS.iloc[0]

    df_BS = df_BS[1:]

    df_BS.set_index("in million USD", inplace=True)

    df_BS.fillna(0, inplace=True)

    



    df_CF = df.iloc[index_CF-2:, 1:]

    df_CF.dropna(how='all', inplace=True)

    df_CF.columns = df_CF.iloc[0]

    df_CF = df_CF[1:]

    df_CF.set_index("in million USD", inplace=True)

    df_CF.fillna(0, inplace=True)

    

    df_CF = df_CF.T

    df_BS = df_BS.T

    df_PL = df_PL.T

    

    return df, df_PL, df_BS, df_CF



def combine_regexes(regexes):

    return "(" + ")|(".join(regexes) + ")"
for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

_,southwest_PL, southwest_BS, southwest_CF = excel_to_df("/kaggle/input/homework22/SimFin-data (12).xlsx")  
southwest_BS
del(southwest_BS["Assets"])

southwest_BS
southwest_BS["_Total Current Assets"] = southwest_BS["Cash and cash equivalents"] + southwest_BS["Deferred income taxes1"] + southwest_BS["Inventories of parts and supplies, at cost"]  + southwest_BS["Short-term investments"] + southwest_BS["Prepaid expenses and other current assets"] + southwest_BS["Deferred income taxes2"] + southwest_BS["Accounts and other receivables"]
import pandas as pd

Southwest= pd.read_csv("../input/homework27/Southwest Air Line.csv")
southwest_BS["_NonCurrent Assets"] = southwest_BS["Total assets"] - southwest_BS["_Total Current Assets"] 
southwest_BS["Total liabilities"] = southwest_BS["Total liabilities and stockholders' equity"] - southwest_BS["Total stockholders' equity"]
southwest_BS["Total NonCurrent Liabilities"] = southwest_BS["Total liabilities"] - southwest_BS["Total current liabilities"]
southwest_BS["Total Current Ratio"] = southwest_BS["_Total Current Assets"] / southwest_BS["Total current liabilities"]
southwest_BS["Total Current Ratio"] = southwest_BS["_Total Current Assets"] / southwest_BS["Total current liabilities"]
southwest_BS["Net Asset Value"] = (southwest_BS["Total liabilities and stockholders' equity"] - southwest_BS["Total liabilities"]) / 546
southwest_BS["Net Asset Value"]. plot()
southwest_BS["Total Equity"] = southwest_BS["Total stockholders' equity"]
southwest_BS
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
import chart_studio

# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 

# https://chart-studio.plot.ly/feed/#/



# Un-remark the code below and add your own your_username and own your_apikey

chart_studio.tools.set_credentials_file(username='', api_key='')
import plotly

import plotly.graph_objs as go

!pip install plotly --upgrade

import plotly.offline as py

from plotly.graph_objs import Scatter, Layout

import plotly.graph_objs as go



py.init_notebook_mode(connected=True)
good_stuff = '''

Cash and cash equivalents

Short-term investments

Deferred income taxes1

Deferred income taxes2

Accounts and other receivables

Prepaid expenses and other current assets

Inventories of parts and supplies, at cost

'''



asset_columns = [ x for x in good_stuff.strip().split("\n") ]
asset_columns
southwest_BS[ asset_columns ].plot()
good_stuff2 = '''

Total liabilities

Total assets

Total Current Ratio

Total Equity



'''



asset_columns2 = [ x for x in good_stuff2.strip().split("\n") ]
asset_columns2
southwest_BS[ asset_columns2 ].plot()
assets = go.Bar(

    x=southwest_BS.index,

    y=southwest_BS["Total assets"],

    name='Assets'

)

libailities = go.Bar(

    x=southwest_BS.index,

    y=southwest_BS["Total liabilities"],

    name='Libailities'

)



shareholder_equity = go.Scatter(

    x=southwest_BS.index,

    y=southwest_BS["Total Equity"],

    name='Equity'

)



data = [assets, libailities, shareholder_equity]

layout = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

py.iplot(fig_bs, filename='Total Assets and Liabilities')
southwest_BS["book value"] = southwest_BS["Total Equity"] / 546
southwest_BS["book value"]
southwest_BS["price to book value"] = southwest_CF["stock price"] / southwest_BS["book value"]
southwest_BS["working capital"] = southwest_BS["_Total Current Assets"] - southwest_BS["Total current liabilities"]
southwest_BS[["working capital"]].plot()
southwest_PL["Price to Equity"] = southwest_CF["stock price"] / southwest_BS["Total Equity"] 
southwest_PL["Price to Equity per share"] = southwest_PL["Price to Equity"] / 546
southwest_BS["Total Debt"] = southwest_BS["Long-term debt less current maturities"] + southwest_BS["Current maturities of long-term debt"] + southwest_BS["Current maturities of long-term debt1"]
southwest_BS["Debt to Equity Ratio"] =southwest_BS["Total Debt"] / southwest_BS["Total Equity"]
southwest_BS["working capital"] = southwest_BS["_Total Current Assets"] - southwest_BS["Total current liabilities"]
southwest_BS[["working capital"]].plot()
southwest_BS["Average Equity Growth"] = (9853 - 5454) / 10
southwest_PL["Price to Equity growth"] = southwest_PL["Price to Equity"] / southwest_BS["Average Equity Growth"]
southwest_PL["EBIT"] = southwest_PL["Operating Income"]

southwest_PL["Interest Coverage Ratio"] = southwest_PL["EBIT"] / southwest_PL["Interest expense"]
southwest_PL["Average Basic Shares Outstanding"] = 546

southwest_PL["Average Diluted Shares Outstanding"] = 547.25
southwest_PL["Basic EPS"] = southwest_PL["Net income"] / southwest_PL["Average Basic Shares Outstanding"]

southwest_PL["Diluted EPS"] = southwest_PL["Net income"] / southwest_PL["Average Diluted Shares Outstanding"]
southwest_PL["Net income"] .plot()
southwest_PL
southwest_PL["Basic EPS Growth"] = (4.514652/0.181319)**(1/10) - 1



southwest_PL["Diluted EPS Growth"] = (4.504340/0.180905)**(1/10) - 1
southwest_PL["Basic EPS 10 years from now"] = (1 + southwest_PL["Basic EPS Growth"] )**10*4.514652

southwest_PL["Diluted EPS 10 years from now"] = (1+ southwest_PL["Diluted EPS Growth"] )**10*4.504340
southwest_PL["Price to Earnings ratio"] = southwest_CF["stock price"] / southwest_PL["Net income"] * southwest_PL["Average Basic Shares Outstanding"]
southwest_PL["Price to Earnings ratio"]. plot()
southwest_PL["Average Price to Earnings ratio"] = sum(southwest_PL["Price to Earnings ratio"]) /10
southwest_PL["Future Price of southwest Air Line"] = southwest_PL["Average Price to Earnings ratio"] * southwest_PL["Diluted EPS 10 years from now"]
southwest_PL["Target buy Price Today"] = southwest_PL["Future Price of southwest Air Line"] / (1 + southwest_PL["Basic EPS Growth"])
southwest_PL["Current Target Buy Price"] = southwest_PL["Diluted EPS"] * southwest_PL["Price to Earnings ratio"]
southwest_PL["Margin of safety"] = southwest_PL["Current Target Buy Price"] *(1-0.25)

good_stuff2 = '''

Current Target Buy Price

Price to Earnings ratio

Interest Coverage Ratio

Price to Equity growth

Margin of safety

'''



asset_columns6 = [ x for x in good_stuff2.strip().split("\n") ]
asset_columns6
southwest_PL[ asset_columns6 ].plot()
southwest_PL["Interest Coverage Ratio"]. plot()
good_stuff2 = '''

price to book value

Total stockholders' equity

'''
asset_columns7 = [ x for x in good_stuff2.strip().split("\n") ]
southwest_BS[["book value" , "price to book value" ]]. plot()



southwest_BS["Total stockholders' equity"]. plot()
southwest_BS["book value"]. plot()

southwest_BS["book value per share"] = southwest_BS["book value"] / 546 

southwest_BS["Total equity per share"] = southwest_BS["Total Equity"] / 546

southwest_BS["Net asset value per share"] = southwest_BS["Net Asset Value"] /546
bvp = go.Scatter(

    x=southwest_BS.index,

    y=southwest_BS["Total assets"],

    name='bvp'

)

tvp = go.Scatter(

    x=southwest_BS.index,

    y=southwest_BS["Total liabilities"],

    name='tvp'

)



nvp = go.Scatter(

    x=southwest_BS.index,

    y=southwest_BS["Total Equity"],

    name='nvp'

)



data = [bvp, tvp, nvp]

layout = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

py.iplot(fig_bs, filename='break-up value')