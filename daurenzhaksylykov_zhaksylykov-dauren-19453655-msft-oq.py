#Zhaksylykov Dauren 19453655
#Balance Sheet & Cash Flow Analysis
#Value Investing Stock Analysis with Python
#Source: https://github.com/taewookim/YouTube
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
import os
#import tk_library
#from tk_library import excel_to_df
#!pip install plotly==4.5.4
#!pip install chart_studio
#!pip install xlrd
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_, microsoft_PL, microsoft_BS, microsoft_CF = excel_to_df("/kaggle/input/microsoft-fin-stat/Microsoft_Fin_Stat.xlsx")
microsoft_BS
del(microsoft_BS['Assets'])
microsoft_BS
microsoft_BS["_Total Current Assets"] = microsoft_BS["Cash, Cash Equivalents & Short Term Investments"] + microsoft_BS["Accounts & Notes Receivable"] + microsoft_BS["Inventories"] + microsoft_BS["Other Short Term Assets"]
microsoft_BS[["_Total Current Assets", "Total Current Assets"]]
microsoft_BS["_NonCurrent Assets"] = microsoft_BS["Property, Plant & Equipment, Net"] + microsoft_BS["Long Term Investments & Receivables"] + microsoft_BS["Other Long Term Assets"]
microsoft_BS["_Total Assets"] = microsoft_BS["_NonCurrent Assets"] + microsoft_BS["_Total Current Assets"]
microsoft_BS["_Total Liabilities"] = microsoft_BS["Total Current Liabilities"] + microsoft_BS["Total Noncurrent Liabilities"]
microsoft_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline
microsoft_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Other Short Term Assets
'''

asset_columns = [ x for x in good_stuff.strip().split("\n") ]
asset_columns
microsoft_BS[ asset_columns ].plot()
import chart_studio
chart_studio.tools.set_credentials_file(username='daurenzhaks', api_key='TkIURYSWumG8BEjBQsWk')
import chart_studio.plotly as py
import plotly.graph_objs as go
#from tk_library import combine_regexes
assets = go.Bar(
    x=microsoft_BS.index,
    y=microsoft_BS["Total Assets"],
    name='Assets'
)
liabilities = go.Bar(
    x=microsoft_BS.index,
    y=microsoft_BS["Total Liabilities"],
    name='Liabilities'
)

shareholder_equity = go.Scatter(
    x=microsoft_BS.index,
    y=microsoft_BS["Total Equity"],
    name='Equity'
)

data = [assets, liabilities, shareholder_equity]
layout = go.Layout(
    barmode='stack'
)

fig_bs = go.Figure(data=data, layout=layout)

fig_bs.show() 
asset_data = []
columns = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Other Short Term Assets
Property, Plant & Equipment, Net
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    asset_bar = go.Bar(
        x=microsoft_BS.index,
        y=microsoft_BS[ col ],
        name=col
    )    
    asset_data.append(asset_bar)
    
layout_assets = go.Layout(
    barmode='stack'
)

fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)

fig_bs_assets.show()
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
        x=microsoft_BS.index,
        y=microsoft_BS[ col ],
        name=col
    )    
    liability_data.append(liability_bar)
    
layout_liabilitys = go.Layout(
    barmode='stack'
)

fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)

fig_bs_liabilitys.show() 
microsoft_BS["Working Capital"] = microsoft_BS["Total Current Assets"] - microsoft_BS["Total Current Liabilities"]
microsoft_BS[["Working Capital"]].plot()
microsoft_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Using Chart Studio in Plotly 

PR_data = []
columns = '''
Accounts & Notes Receivable
Payables & Accruals
'''

for col in columns.strip().split("\n"):
    PR_Scatter = go.Scatter(
        x=microsoft_BS.index,
        y=microsoft_BS[ col ],
        name=col
    )    
    PR_data.append(PR_Scatter)
    
layout_PR = go.Layout(
    barmode='stack'
)

fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)

fig_bs_PR.show() 
microsoft_BS["Inventories"].plot()
microsoft_BS[ ["Property, Plant & Equipment, Net", "Long Term Investments & Receivables", "Other Long Term Assets"] ].plot()
# Using Plotly

AAA_data = []
columns = '''
Property, Plant & Equipment, Net
Long Term Investments & Receivables
Other Long Term Assets
'''

for col in columns.strip().split("\n"):
    AAA_bar = go.Bar(
        x=microsoft_BS.index,
        y=microsoft_BS[ col ],
        name=col
    )    
    AAA_data.append(AAA_bar)
    
layout_AAA = go.Layout(
    barmode='stack'
)


fig_bs_AAA = go.Figure(data=AAA_data, layout=layout_AAA)

fig_bs_AAA.show()
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
        x=microsoft_BS.index,
        y=microsoft_BS[ col ],
        name=col
    )    
    equity_data.append(equity_Scatter)
    
layout_equity = go.Layout(
    barmode='stack'
)

fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)

fig_bs_equity.show() 
# Book Value= Total Assets-Intangible Assets-Liabilities-Preferred Stock Value
# According to simfin data, Microsoft has no preferred stock, no intangible assets, and no goodwill

microsoft_BS["book value"] = microsoft_BS["Total Assets"] - microsoft_BS["Total Liabilities"]
microsoft_BS["book value"].plot()
#Price-to-Earnings Growth Ratio (PEG forward)  
#Formula = Price-to-Earnings Ratio/Earnings-Growth-Rate
#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp

PE_RATIO = 25.34 # According to Simfin: https://simfin.com/data/companies/59265

# According to NASDAQ: https://finbox.com/NASDAQGS:MSFT/explorer/eps_proj_growth
GROWTH_RATE = 0.1360 # Expected growth over the five next years

PEG_ratio = PE_RATIO / (GROWTH_RATE*100)

print("Microsoft Corp's PEG Ratio is", PEG_ratio)
#Step 1: Find EPS Annual Compounded Growth Rate
#The earnings per share information have been collected from: https://www.macrotrends.net/stocks/charts/AAPL/apple/revenue
epsTTM10 = 2.1
epsTTM = 5.06
CAGR = (epsTTM/epsTTM10)**(1/9) - 1
print(CAGR)
#Step 2: Estimate EPS 10 years from now
#The CAGR formula is used to extract epsTTM from it, making it the EPS value 10 years from now.
eps10 = (1+CAGR)**10*epsTTM
print(eps10)
#Step 3: Estimate Stock Price 10 years from now
#Estimated future EPS x Average PE ratio
#To calculate the future price 10 years from now, we multiply future EPS by average PE ratio for the past 5 years, obtained from: https://ycharts.com/companies/MSFT/pe_ratio
aver_pe = 33.13
future_price = eps10 * aver_pe
print(future_price)
#Step 4: To calculate the target buy price, we discount the future price back to now using 6.5% average WACC for computer services companies obtained from http://people.stern.nyu.edu/adamodar/New_Home_Page/datafile/wacc.htm
target_buy = future_price / (1 + 0.065) ** 10
print(target_buy)
mar_safety = target_buy * 0.85
print(mar_safety)
debt_to_equity = microsoft_BS['Total Liabilities'] / microsoft_BS['Total Equity']
print(debt_to_equity)
#Interest expense for 2019 has been obtained from Finance Yahoo:https://finbox.com/NASDAQGS:MSFT/explorer/interest_exp
int_exp=2632000000000
pretax_income=49323000000000
int_cov = pretax_income / int_exp
print(int_cov)
microsoft_PL
microsoft_PL["Net Income Available to Common Shareholders"]
return_on_equity=microsoft_PL["Net Income Available to Common Shareholders"]/microsoft_BS["Total Equity"]
print(return_on_equity)
return_on_equity.plot()
return_on_assets=microsoft_PL["Net Income Available to Common Shareholders"]/microsoft_BS["Total Assets"]
print(return_on_assets)
return_on_assets.plot()