%%time
!pip install simfin
import simfin as sf
from simfin.names import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set your API-key for downloading data. This key gets the free data.
sf.set_api_key(api_key='free')

# Set the local directory where data-files are stored.
# The directory will be created if it does not already exist.
sf.set_data_dir('~/simfin_data/')

# Download the data from the SimFin server and load into a Pandas DataFrame.
df_IS = sf.load(dataset='income',variant='annual', market='us',
              index=[TICKER, REPORT_DATE],
              parse_dates=[REPORT_DATE, PUBLISH_DATE])

df_BS = sf.load(dataset='balance',variant='annual', market='us',
              index=[TICKER, REPORT_DATE],
              parse_dates=[REPORT_DATE, PUBLISH_DATE])

df_CF = sf.load(dataset='cashflow',variant='annual', market='us',
              index=[TICKER, REPORT_DATE],
              parse_dates=[REPORT_DATE, PUBLISH_DATE])
MMM_IS= df_IS.loc['MMM']
MMM_IS
MMM_IS["EPS"] = MMM_IS.loc[:,"Net Income"] / MMM_IS.loc[:,"Shares (Diluted)"]
print(MMM_IS["EPS"])
import matplotlib.pyplot as plt
MMM_IS[['EPS']].plot()
plt.show()
import math
eps2008 = MMM_IS["EPS"][0]
eps2018 = MMM_IS["EPS"][-1]
print(eps2008)
print(eps2018)
# CAGR known as Compound Annual Growth Rate. To calculate it,
    
    # Divide the value of an investment at the end of the period by its value at the beginning of that period.
    # Raise the result to an exponent of one divided by the number of years.
    # Subtract one from the subsequent result.
    
CAGR_EPS = pow((eps2018/eps2008),(1/10)) -1
print(CAGR_EPS)
# Forecast EPS:
    # EPS 10 years over now = Current year EPS*（1 + EPS CAGR）^n
    
EstimateEPS = eps2018* pow(1+CAGR_EPS,10)
print(EstimateEPS)
# Estimate 5 year average PE ratio is calculated by 
    # the average of P/E ratio 5-year high and P/E ratio 5-year low
    # data from MSN Finance

PEratio5YearHigh = 25.66
PEratio5YearLow = 19.86
FiveYearAveragePEratio = (PEratio5YearHigh + PEratio5YearLow) /2
print(FiveYearAveragePEratio)


# Intrinsic value of target share could be calculated by discounting the Market value per share, using WACC as discount rate.
# Five year Average Growth Estimate is 5.2 percent (data from yahoo finance)


MarketValuePerShare = eps2018* pow((1+0.052),5) * FiveYearAveragePEratio

# WACC = (Equity / Total Capital) * CoE + (Debt / Total Capital) * CoD * (1 - Tax Rate)
    # Total Capital = Debt + Equity
    #CoD = Cost of Debt
    #CoE = Cost of Equity
    
# 3M's wacc is 9.0% (data from finbox.com)

IntrinsicValue = MarketValuePerShare / pow(1+0.09,4)
IntrinsicValue
Margin_Safety = 0.25

TargetBuyPrice = IntrinsicValue * (1 - Margin_Safety)
TargetBuyPrice

MMM_BS=df_BS.loc["MMM"]
MMM_BS
MMM_BS["DebtEquityRatio"] = MMM_BS.loc[:,"Total Liabilities"] /(MMM_BS.loc[:,"Total Liabilities & Equity"] - MMM_BS.loc[:,"Total Liabilities"])
MMM_BS["DebtEquityRatio"]
MMM_BS[["DebtEquityRatio"]].plot()
plt.show()
 # EBIT = NET Income + Interest Expense + Income Tax
MMM_IS["EBIT"] = MMM_IS.loc[:,"Net Income"] - MMM_IS.loc[:,"Interest Expense, Net"] - MMM_IS.loc[:,"Income Tax (Expense) Benefit, Net"]
MMM_IS["InterestCoverageRatio"]= MMM_IS["EBIT"] / MMM_IS.loc[:,"Interest Expense, Net"] *(-1)
MMM_IS["InterestCoverageRatio"]

MMM_IS[["InterestCoverageRatio"]].plot()
plt.show()
MMM_CF=df_CF.loc['MMM']
print('\nThe List of Balance Sheet Columns:')
[print(x) for x in MMM_BS.columns]
#!pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe

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
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_, df_MMM_PL, df_MMM_BS, df_MMM_CF = excel_to_df("/kaggle/input/3mdata/3m data.xlsx")
del(df_MMM_BS["Assets"])

print('\nThe List of Balance Sheet Columns:')
[print(x) for x in df_MMM_BS.columns]
df_MMM_BS["_Total Current Assets"] = df_MMM_BS["Cash, Cash Equivalents & Short Term Investments"] + df_MMM_BS[
         "Accounts & Notes Receivable"] + df_MMM_BS["Inventories"] + df_MMM_BS["Other Short Term Assets"]

df_MMM_BS[["_Total Current Assets", "Total Current Assets"]]

good_stuff = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Other Short Term Assets
'''
asset_columns = [ x for x in good_stuff.strip().split("\n") ]

df_MMM_BS[ asset_columns ].plot()

df_MMM_BS["_NonCurrent Assets"] = df_MMM_BS["Property, Plant & Equipment, Net"] + df_MMM_BS[
         "Long Term Investments & Receivables"] + df_MMM_BS["Other Long Term Assets"]

df_MMM_BS["_Total Assets"] = df_MMM_BS["_NonCurrent Assets"] + df_MMM_BS["_Total Current Assets"] 
df_MMM_BS[["_Total Assets", "Total Assets"]]
df_MMM_BS["_Total Liabilities"] = df_MMM_BS["Total Current Liabilities"] + df_MMM_BS["Total Noncurrent Liabilities"]
df_MMM_BS[["_Total Liabilities", "Total Liabilities"]]

!pip install chart_studio
import chart_studio
import chart_studio.plotly as py
import plotly.graph_objs as go
chart_studio.tools.set_credentials_file(username='jwfeng10', api_key='SwjatfeB708ylI2581Eu')

assets = go.Bar(

x=df_MMM_BS.index,
y=df_MMM_BS["Total Assets"],
name='Assets'
)
liabilities = go.Bar(
x=df_MMM_BS.index,
y=df_MMM_BS["Total Liabilities"],
name='Liabilities'
)

shareholder_equity = go.Scatter(
x=df_MMM_BS.index,
y=df_MMM_BS["Total Equity"],
name='Equity'
)

data = [assets, liabilities, shareholder_equity]
layout = go.Layout(
barmode='stack'
)

fig_bs = go.Figure(data=data, layout=layout)
py.iplot(fig_bs, filename='Total Assets and Liabilities')
import chart_studio.plotly as py
import plotly.graph_objs as go
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
        x=df_MMM_BS.index,
        y=df_MMM_BS[ col ],
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
        x=df_MMM_BS.index,
        y=df_MMM_BS[ col ],
        name=col
    )    
    liability_data.append(liability_bar)
    
layout_liabilitys = go.Layout(
    barmode='stack'
)

fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)
py.iplot(fig_bs_liabilitys, filename='Total liabilities Breakdown')
df_MMM_BS[['Total Assets', 'Total Liabilities', 'Total Equity']].plot()
plt.show()
df_MMM_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
plt.show()
df_MMM_BS["Inventories"].plot()
plt.show()
df_MMM_BS[ ["Property, Plant & Equipment, Net", "Long Term Investments & Receivables", "Other Long Term Assets"] ].plot()
plt.show()
LongTermAsset_data = []
columns = '''
Property, Plant & Equipment, Net
Long Term Investments & Receivables
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    LongTermAsset_bar = go.Bar(
        x=df_MMM_BS.index,
        y=df_MMM_BS[ col ],
        name=col
    )    
    LongTermAsset_data.append(LongTermAsset_bar)
    
layout_LongTermAsset = go.Layout(
    barmode='stack'
)

fig_bs_LongTermAsset = go.Figure(data=LongTermAsset_data, layout=layout_LongTermAsset)
py.iplot(fig_bs_LongTermAsset, filename='Total Long Term Assets')
equity_columns = '''
Share Capital & Additional Paid-In Capital
Treasury Stock
Retained Earnings
Other Equity
Equity Before Minority Interest
Minority Interest
'''

equity_columns = [ x for x in equity_columns.strip().split("\n")]
df_MMM_BS[ equity_columns ].plot()
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
        x=df_MMM_BS.index,
        y=df_MMM_BS[ col ],
        name=col
    )    
    equity_data.append(equity_Scatter)
    
layout_equity = go.Layout(
    barmode='stack'
)

fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)
py.iplot(fig_bs_equity, filename='Total Equity')
df_MMM_BS["Book Value"] = df_MMM_BS["Total Assets"] - df_MMM_BS["Total Liabilities"]
df_MMM_BS[["Book Value"]].plot()
plt.show()
df_MMM_BS["working capital"] = df_MMM_BS["Total Current Assets"] - df_MMM_BS["Total Current Liabilities"]
df_MMM_BS[["working capital"]].plot()

print('\nThe List of Profit&Losses Columns:')
[print(x) for x in df_MMM_PL.columns]
PL_data = []
columns = '''
Revenue
Cost of revenue
Gross Profit
Operating Expenses
Operating Income (Loss)
Non-Operating Income (Loss)
Pretax Income (Loss), Adjusted
Abnormal Gains (Losses)
Pretax Income (Loss)
Income Tax (Expense) Benefit, net
Income (Loss) Including Minority Interest
Minority Interest
Net Income Available to Common Shareholders
'''


for col in columns.strip().split("\n"):
    PL_Scatter = go.Scatter(
        x=df_MMM_PL.index,
        y=df_MMM_PL[ col ],
        name=col
    )    
    PL_data.append(PL_Scatter)
    
layout_PL = go.Layout(
    barmode='stack'
)

fig_bs_PL = go.Figure(data=PL_data, layout=layout_PL)
py.iplot(fig_bs_PL, filename='profit&losses')
expense_data = []
columns = '''
Revenue
Cost of revenue
'''


for col in columns.strip().split("\n"):
    PL_Scatter = go.Scatter(
        x=df_MMM_PL.index,
        y=df_MMM_PL[ col ],
        name=col
    )    
    expense_data.append(PL_Scatter)
    
layout_PL = go.Layout(
    barmode='stack'
)

fig_bs_PL = go.Figure(data=expense_data, layout=layout_PL)
py.iplot(fig_bs_PL, filename='profit&losses')
print('\nThe List of Cash flow Columns:')
[print(x) for x in df_MMM_CF.columns]
CF_data = []
columns = '''

Net Income/Starting Line
Cash from Operating Activities
Net Changes in Cash
'''


for col in columns.strip().split("\n"):
    CF_Scatter = go.Scatter(
        x=df_MMM_CF.index,
        y=df_MMM_CF[ col ],
        name=col
    )    
    CF_data.append(CF_Scatter)
    
layout_CF = go.Layout(
    barmode='stack'
)

fig_bs_CF = go.Figure(data=CF_data, layout=layout_CF)
py.iplot(fig_bs_CF, filename='cashflow')