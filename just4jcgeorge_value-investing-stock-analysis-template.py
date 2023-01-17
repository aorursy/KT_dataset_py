# Do the below in kaggle

!pip install plotly==4.4.1

!pip install chart_studio

!pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe
# Add 'tk_library.py' file given by your tutor, as a utility script under 'File'

# Look for it under usr/bin on the right drawer



# import excel_to_df function

import os

import tk_library_py

from tk_library_py import excel_to_df
# Show the files and their pathnames

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Add your simfin-data.xlsx using the '+ Add Data' top right button

_, exxon_PL, exxon_BS, exxon_CF = excel_to_df("/kaggle/input/simfin-data/SimFin-data.xlsx")
exxon_BS
del(exxon_BS["Assets"])

exxon_BS
exxon_BS["_Total Current Assets"] = exxon_BS["Cash, Cash Equivalents & Short Term Investments"] + exxon_BS["Accounts & Notes Receivable"] + exxon_BS["Inventories"] + exxon_BS["Other Short Term Assets"]
exxon_BS[["_Total Current Assets", "Total Current Assets"]]
exxon_BS["_NonCurrent Assets"] = exxon_BS["Property, Plant & Equipment, Net"] + exxon_BS["Long Term Investments & Receivables"] + exxon_BS["Other Long Term Assets"]
exxon_BS["_Total Assets"] = exxon_BS["_NonCurrent Assets"] + exxon_BS["_Total Current Assets"] 
exxon_BS["_Total Liabilities"] = exxon_BS["Total Current Liabilities"] + exxon_BS["Total Noncurrent Liabilities"]
exxon_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline

exxon_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''



asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
exxon_BS[ asset_columns ].plot()
import chart_studio

# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 

# https://chart-studio.plot.ly/feed/#/



# Un-remark the code below and add your own your_username and own your_apikey

chart_studio.tools.set_credentials_file(username='', api_key='')
import chart_studio.plotly as py

import plotly.graph_objs as go

from tk_library_py import combine_regexes

assets = go.Bar(

    x=exxon_BS.index,

    y=exxon_BS["Total Assets"],

    name='Assets'

)

liabilities = go.Bar(

    x=exxon_BS.index,

    y=exxon_BS["Total Liabilities"],

    name='Liabilities'

)



shareholder_equity = go.Scatter(

    x=exxon_BS.index,

    y=exxon_BS["Total Equity"],

    name='Equity'

)



data = [assets, liabilities, shareholder_equity]

layout = go.Layout(

    barmode='stack'

)



fig_bs = go.Figure(data=data, layout=layout)

py.iplot(fig_bs, filename='Total Assets and Liabilities')
asset_data = []

columns = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

Property, Plant & Equipment, Net

Long Term Investments & Receivables

Other Long Term Assets

'''





for col in columns.strip().split("\n"):

    asset_bar = go.Bar(

        x=exxon_BS.index,

        y=exxon_BS[ col ],

        name=col

    )    

    asset_data.append(asset_bar)

    

layout_assets = go.Layout(

    barmode='stack'

)



fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)

py.iplot(fig_bs_assets, filename='Total Assets Breakdown')
liability_data = []

columns = '''

Payables & Accruals

Short Term Debt

Other Short Term Liabilities

Long Term Debt

Other Long Term Liabilities

'''





for col in columns.strip().split("\n"):

    liability_bar = go.Bar(

        x=exxon_BS.index,

        y=exxon_BS[ col ],

        name=col

    )    

    liability_data.append(liability_bar)

    

layout_liabilitys = go.Layout(

    barmode='stack'

)



fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)

py.iplot(fig_bs_liabilitys, filename='Total liabilities Breakdown')
exxon_BS["working capital"] = exxon_BS["Total Current Assets"] - exxon_BS["Total Current Liabilities"]
exxon_BS[["working capital"]].plot()
exxon_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Using Chart Studio in Plotly 



PR_data = []

columns = '''

Accounts & Notes Receivable

Payables & Accruals

'''



for col in columns.strip().split("\n"):

    PR_Scatter = go.Scatter(

        x=exxon_BS.index,

        y=exxon_BS[ col ],

        name=col

    )    

    PR_data.append(PR_Scatter)

    

layout_PR = go.Layout(

    barmode='stack'

)



fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)

py.iplot(fig_bs_PR, filename='Accounts & Notes Receivable vs Payables & Accruals')
exxon_BS["Inventories"].plot()
exxon_BS[ ["Property, Plant & Equipment, Net", "Long Term Investments & Receivables", "Other Long Term Assets"] ].plot()
# Using Plotly



AAA_data = []

columns = '''

Property, Plant & Equipment, Net

Long Term Investments & Receivables

Other Long Term Assets

'''





for col in columns.strip().split("\n"):

    AAA_bar = go.Bar(

        x=exxon_BS.index,

        y=exxon_BS[ col ],

        name=col

    )    

    AAA_data.append(AAA_bar)

    

layout_AAA = go.Layout(

    barmode='stack'

)



fig_bs_AAA = go.Figure(data=AAA_data, layout=layout_AAA)

py.iplot(fig_bs_AAA, filename='Total Long Term Assets')
equity_columns = '''

Share Capital & Additional Paid-In Capital

Treasury Stock

Retained Earnings

Other Equity

Equity Before Minority Interest

Minority Interest

'''



equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
exxon_BS[ equity_columns ].plot()
# Using Plotly



equity_data = []

columns = '''

Share Capital & Additional Paid-In Capital

Treasury Stock

Retained Earnings

Other Equity

Equity Before Minority Interest

Minority Interest

'''





for col in columns.strip().split("\n"):

    equity_Scatter = go.Scatter(

        x=exxon_BS.index,

        y=exxon_BS[ col ],

        name=col

    )    

    equity_data.append(equity_Scatter)

    

layout_equity = go.Layout(

    barmode='stack'

)



fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)

py.iplot(fig_bs_equity, filename='Total Equity')
# Exxon mobil has no preferred stock, no intengible assets, and no goodwill



exxon_BS["book value"] = exxon_BS["Total Assets"] - exxon_BS["Total Liabilities"]

exxon_BS["book value"].plot()
exxon_BS["current ratio"] = exxon_BS["Total Current Assets"] / exxon_BS["Total Current Liabilities"]
exxon_BS["current ratio"].plot()
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  

#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate

#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp



PE_RATIO = 16.38 # FROM SIMFIN WEBSITE: https://simfin.com/data/companies/121214



# FROM NASDAQ WEBSITE: https://www.nasdaq.com/symbol/xom/earnings-growth

GROWTH_RATE = 0.1394 # Forcast over the five next years



PEG_ratio = PE_RATIO / (GROWTH_RATE*100)



print("EXXON Mobil Corp's PEG Ratio is", PEG_ratio)
#End of Value Investing Stock Analysis Template