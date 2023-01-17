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
# Add 'tk_library.py' file given by your tutor, as a utility script under 'File'
# Look for it under usr/bin on the right drawer

# import excel_to_df function
import os
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
# Add your simfin-data.xlsx using the '+ Add Data' top right button
_,delta_PL, delta_BS, delta_CF = excel_to_df("/kaggle/input/homework2/SimFin-data (7).xlsx")                                  
delta_BS
del(delta_BS["Assets"])
delta_BS["_Total Current Assets"] = delta_BS["Cash and cash equivalents"] + delta_BS["Accounts receivable, net of an allowance for uncollectible accounts"] + delta_BS["Fuel inventory"]  + delta_BS["Restricted cash, cash equivalents and short-term investments"] + delta_BS["Prepaid expenses and other"] + delta_BS["Hedge margin receivable"] + delta_BS["Short-term investments"] + delta_BS["Expendable parts and supplies inventories, net of an allowance for obsolescence"] + delta_BS["Hedge derivative asset"]
delta_BS
delta_BS[["_Total Current Assets", "_Total Current Assets"]]
delta_BS["_NonCurrent Assets"] = delta_BS["Property and equipment, net of accumulated depreciation and amortization"] + delta_BS["Operating lease right-of-use assets"] + delta_BS["Goodwill"] + delta_BS["Identifiable intangibles, net of accumulated amortization"] + delta_BS["Cash restricted for airport construction"] + delta_BS["OTHER NONCURRENT ASSETS"] 
delta_BS["_Total Assets"] = delta_BS["_NonCurrent Assets"] + delta_BS["_Total Current Assets"] 
delta_BS["_Total Liabilities"] = delta_BS["Total liabilities and stockholders' equity"] - delta_BS["Total stockholders' equity"]
delta_BS["_Total Current Liabilities"] = delta_BS["_Total Liabilities"] - delta_BS["Total Noncurrent Liabilities"]
delta_BS["_Total Current Ratio"] = delta_BS["_Total Current Assets"] / delta_BS["_Total Current Liabilities"]
delta_BS["Net Asset Value"] = (delta_BS["Total liabilities and stockholders' equity"] - delta_BS["_Total Liabilities"]) / (663.25*4)
delta_BS["_Total Equity"] = delta_BS["Total stockholders' equity"]
delta_BS
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
good_stuff = '''
Cash and cash equivalents
Short-term investments
Restricted cash, cash equivalents and short-term investments
Accounts receivable, net of an allowance for uncollectible accounts
Hedge margin receivable
Fuel inventory
Restricted cash, cash equivalents and short-term investments
'''

asset_columns = [ x for x in good_stuff.strip().split("\n") ]


asset_columns
delta_BS[ asset_columns ].plot()
good_stuff2 = '''
_Total Liabilities
_Total Assets
_Total Current Ratio
_Total Equity

'''

asset_columns2 = [ x for x in good_stuff2.strip().split("\n") ]
asset_columns2
delta_BS[ asset_columns2 ].plot()
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

delta_BS
assets = go.Bar(
    x=delta_BS.index,
    y=delta_BS["_Total Assets"],
    name='Assets'
)
libailities = go.Bar(
    x=delta_BS.index,
    y=delta_BS["_Total Liabilities"],
    name='Libailities'
)

shareholder_equity = go.Scatter(
    x=delta_BS.index,
    y=delta_BS["_Total Equity"],
    name='Equity'
)

data = [assets, libailities, shareholder_equity]
layout = go.Layout(
    barmode='stack'
)

fig_bs = go.Figure(data=data, layout=layout)
py.iplot(fig_bs, filename='Total Assets and Liabilities')
delta_BS
asset_data = []
columns = '''
Cash and cash equivalents
Short-term investments
Restricted cash, cash equivalents and short-term investments
Accounts receivable, net of an allowance for uncollectible accounts
Hedge margin receivable
Fuel inventory
Restricted cash, cash equivalents and short-term investments
Hedge derivative asset
Expendable parts and supplies inventories, net of an allowance for obsolescence
Deferred income taxes, net
Prepaid expenses and other
Retained earnings
Accumulated other comprehensive loss
Treasury stock, at cost
'''


for col in columns.strip().split("\n"):
    asset_bar = go.Bar(
        x=delta_BS.index,
        y=delta_BS[ col ],
        name=col
    )    
    asset_data.append(asset_bar)
    
layout_assets = go.Layout(
    barmode='stack'
)

fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)
py.iplot(fig_bs_assets, filename='Total Assets Breakdown')
delta_BS["working capital"] = delta_BS["_Total Current Assets"] - delta_BS["_Total Current Liabilities"]
delta_BS[["working capital"]].plot()
delta_CF
delta_PL
delta_BS["book value"] = delta_BS["Total stockholders' equity"] / 660.25
delta_BS
delta_PL["Price to Equity"] = delta_PL["Net income"] / delta_BS["_Total Equity"]
delta_BS ["Average Equity Growth"] = (13687-245) / 10
delta_PL
delta_PL["Price to Equity Growth"] = delta_PL["Price to Equity"] / delta_BS["Average Equity Growth"]
delta_PL["Taxes"] = delta_CF["Deferred income taxes"] +  delta_PL["Aircraft fuel and related taxes"] + delta_BS["Taxes payable"] 
delta_PL["Interest Expense"] = delta_PL["Interest expense, net"] + delta_CF["Amortization of debt discount, net"]
delta_PL["EBIT"] = delta_CF["Net income"] + delta_PL["Taxes"] + delta_PL["Interest Expense"] + delta_PL["Income tax provision"]
delta_PL["Interest Coverage Ratio"] = delta_PL["EBIT"] / delta_PL["Interest Expense"]
delta_PL["Total Debt"] = delta_BS["Long-term debt and capital leases"] + delta_CF["Extinguishment of debt"]  
delta_PL["Debt to Equity Ratio"] = delta_PL["Total Debt"] / delta_BS["_Total Equity"]
delta_PL["Average Basic Shares Outstanding"] = 660.25 
delta_PL["Average Diluted Shares Outstanding"] = 663.25
delta_PL["Basic EPS"] = delta_PL["Net income"] / delta_PL["Average Basic Shares Outstanding"]
delta_PL["Diluted EPS"] = delta_PL["Net income"] / delta_PL["Average Diluted Shares Outstanding"]
delta_PL["Basic EPS Growth"] = (5.959864/0.898145)**(1/9) - 1

delta_PL["Diluted EPS Growth"] = (5.932906/0.894082)**(1/9) - 1

delta_PL["Diluted EPS"]
delta_PL["Basic EPS 10 years from now"] = (1 + delta_PL["Basic EPS Growth"] )**10*5.959864
delta_PL["Diluted EPS 10 years from now"] = (1+ delta_PL["Diluted EPS Growth"] )**10*5.932906
delta_PL["Price to Earnings ratio"] = 56.71 / delta_PL["Net income"] * 660.25
delta_PL["Average Price to Earnings ratio"] = sum(delta_PL["Price to Earnings ratio"]) /10
delta_PL["Future Price of Delta Air Line"] = delta_PL["Average Price to Earnings ratio"] * delta_PL["Diluted EPS 10 years from now"]
delta_PL["Target buy Price Today"] = delta_PL["Future Price of Delta Air Line"] / (1 + delta_PL["Basic EPS Growth"])
delta_PL["Current Target Buy Price"] = delta_PL["Diluted EPS"] * delta_PL["Price to Earnings ratio"]
delta_PL["Margin of safety"] = delta_PL["Current Target Buy Price"] *(1-0.25)
delta_PL
good_stuff2 = '''
Current Target Buy Price
Price to Earnings ratio
Interest Coverage Ratio
Price to Equity Growth
'''

asset_columns6 = [ x for x in good_stuff2.strip().split("\n") ]
asset_columns6
delta_PL[ asset_columns6 ].plot()
delta_PL["Interest Coverage Ratio"]
good_stuff2 = '''
book value
Total stockholders' equity
'''

asset_columns7 = [ x for x in good_stuff2.strip().split("\n") ]
delta_BS[asset_columns7].plot()