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
# Do the below in kaggle

# !pip install plotly==4.4.1

# !pip install chart_studio

# !pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe
# Add 'tk_library.py' file given by your tutor, as a utility script under 'File'

# Look for it under usr/bin on the right drawer



# import excel_to_df function

import os

# import tk_library_py

# from tk_library_py import excel_to_df
# Show the files and their pathnames

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Add your simfin-data.xlsx using the '+ Add Data' top right button

_, apple_PL, apple_BS, apple_CF = excel_to_df("/kaggle/input/simfindata2/SimFin-data.xlsx")
apple_BS
del(apple_BS["Assets"])

apple_BS
apple_BS["_Total Current Assets"] = apple_BS["Cash, Cash Equivalents & Short Term Investments"] + apple_BS["Accounts & Notes Receivable"] + apple_BS["Inventories"] + apple_BS["Other Short Term Assets"]
apple_BS[["_Total Current Assets", "Total Current Assets"]]
apple_BS["_NonCurrent Assets"] = apple_BS["Property, Plant & Equipment, Net"] + apple_BS["Long Term Investments & Receivables"] + apple_BS["Other Long Term Assets"]
apple_BS["_Total Assets"] = apple_BS["_NonCurrent Assets"] + apple_BS["_Total Current Assets"] 
apple_BS["_Total Liabilities"] = apple_BS["Total Current Liabilities"] + apple_BS["Total Noncurrent Liabilities"]
apple_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline

apple_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''



asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
apple_BS[ asset_columns ].plot()
import chart_studio

# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 

# https://chart-studio.plot.ly/feed/#/



# Un-remark the code below and add your own your_username and own your_apikey

chart_studio.tools.set_credentials_file(username='19450710', api_key='STgOq5JaJRGK8LjUW0TD')
import chart_studio.plotly as py

import plotly.graph_objs as go

# from tk_library import combine_regexes

assets = go.Bar(

    x=apple_BS.index,

    y=apple_BS["Total Assets"],

    name='Assets'

)

liabilities = go.Bar(

    x=apple_BS.index,

    y=apple_BS["Total Liabilities"],

    name='Liabilities'

)



shareholder_equity = go.Scatter(

    x=apple_BS.index,

    y=apple_BS["Total Equity"],

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

        x=apple_BS.index,

        y=apple_BS[ col ],

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

        x=apple_BS.index,

        y=apple_BS[ col ],

        name=col

    )    

    liability_data.append(liability_bar)

    

layout_liabilitys = go.Layout(

    barmode='stack'

)



fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)

py.iplot(fig_bs_liabilitys, filename='Total liabilities Breakdown')
apple_BS["working capital"] = apple_BS["Total Current Assets"] - apple_BS["Total Current Liabilities"]
apple_BS[["working capital"]].plot()
apple_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Using Chart Studio in Plotly 



PR_data = []

columns = '''

Accounts & Notes Receivable

Payables & Accruals

'''



for col in columns.strip().split("\n"):

    PR_Scatter = go.Scatter(

        x=apple_BS.index,

        y=apple_BS[ col ],

        name=col

    )    

    PR_data.append(PR_Scatter)

    

layout_PR = go.Layout(

    barmode='stack'

)



fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)

py.iplot(fig_bs_PR, filename='Accounts & Notes Receivable vs Payables & Accruals')
apple_BS["Inventories"].plot()
apple_BS[ ["Property, Plant & Equipment, Net", "Long Term Investments & Receivables", "Other Long Term Assets"] ].plot()
# Using Plotly



AAA_data = []

columns = '''

Property, Plant & Equipment, Net

Long Term Investments & Receivables

Other Long Term Assets

'''





for col in columns.strip().split("\n"):

    AAA_bar = go.Bar(

        x=apple_BS.index,

        y=apple_BS[ col ],

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

Retained Earnings

Other Equity

Equity Before Minority Interest

'''



equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
apple_BS[ equity_columns ].plot()
# Using Plotly



equity_data = []

columns = '''

Share Capital & Additional Paid-In Capital

Retained Earnings

Other Equity

Equity Before Minority Interest

'''





for col in columns.strip().split("\n"):

    equity_Scatter = go.Scatter(

        x=apple_BS.index,

        y=apple_BS[ col ],

        name=col

    )    

    equity_data.append(equity_Scatter)

    

layout_equity = go.Layout(

    barmode='stack'

)



fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)

py.iplot(fig_bs_equity, filename='Total Equity')
apple_BS["book value"] = apple_BS["Total Assets"] - apple_BS["Total Liabilities"]

apple_BS["book value"].plot()
apple_BS["current ratio"] = apple_BS["Total Current Assets"] / apple_BS["Total Current Liabilities"]
apple_BS["current ratio"].plot()
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  

#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate

#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp



PE_RATIO = 25.63 # FROM SIMFIN WEBSITE: https://simfin.com/data/companies/121214



# FROM NASDAQ WEBSITE: https://www.nasdaq.com/symbol/xom/earnings-growth

GROWTH_RATE = 0.1211 # Forcast over the five next years



PEG_ratio = PE_RATIO / (GROWTH_RATE*100)



print("Apple's PEG Ratio is", PEG_ratio)
EPS_2019 = 11.89

EPS_2010 = 2.16

Apple_CAGR = 0.2087

EPS_10years = EPS_2019*(1+GROWTH_RATE)**10

print("Apple's EPS 10 years from now is aournd", EPS_10years)



share_price = 320.3

PE_ratio = share_price / EPS_2019

future_price = EPS_10years*PE_ratio

print("Apple's future price 10 years from now will be", future_price)
discount_rate = 0.0779

target_buy_price = 162.2512

safety_margin = 0.25

safety_margin_price = target_buy_price*(1-safety_margin)

print("When adding margin of safety Apple's taget buy price will be", safety_margin_price)
apple_BS["DE ratio"] = apple_BS["Long Term Debt"] / apple_BS["Total Equity"]
apple_BS[["DE ratio","Long Term Debt","Total Equity"]]
EBIT = 66.153

interest_expense = 3.471

interest_coverage_ratio = EBIT / interest_expense

print("apple's interest coverage ratio will be",interest_coverage_ratio)