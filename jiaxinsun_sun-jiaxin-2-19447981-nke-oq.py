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
# Show the files and their pathnames
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Add your simfin-data.xlsx using the '+ Add Data' top right button
_, NKE_PL, NKE_BS, NKE_CF = excel_to_df("/kaggle/input/nikes-financial-statement-with-2-version/SimFin-data_s.xlsx")
NKE_BS
del(NKE_BS["Assets"])
NKE_BS
NKE_BS["_Total Current Assets"] = NKE_BS["Cash, Cash Equivalents & Short Term Investments"] + NKE_BS["Accounts & Notes Receivable"] + NKE_BS["Inventories"] + NKE_BS["Other Short Term Assets"]
NKE_BS[["_Total Current Assets", "Total Current Assets"]]
NKE_BS["_NonCurrent Assets"] = NKE_BS["Property, Plant & Equipment, Net"] + NKE_BS["Other Long Term Assets"]
NKE_BS[["_NonCurrent Assets", "Total Noncurrent Assets"]]
NKE_BS["_Total Assets"] = NKE_BS["_NonCurrent Assets"] + NKE_BS["_Total Current Assets"] 
NKE_BS[["_Total Assets", "Total Assets"]]
NKE_BS["_Total Liabilities"] = NKE_BS["Total Current Liabilities"] + NKE_BS["Total Noncurrent Liabilities"]
NKE_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline
NKE_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Other Short Term Assets
'''

asset_columns = [ x for x in good_stuff.strip().split("\n") ]
asset_columns
NKE_BS[ asset_columns ].plot()
!pip install chart_studio
import chart_studio
# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 
# https://chart-studio.plot.ly/feed/#/

# Un-remark the code below and add your own your_username and own your_apikey
chart_studio.tools.set_credentials_file(username='Jiaxin_Sun', api_key='w6SNLY3TSZyQaCa9dMrz')
import chart_studio.plotly as py
import plotly.graph_objs as go
#from tk_library_py import combine_regexes

assets = go.Bar(
    x=NKE_BS.index,
    y=NKE_BS["Total Assets"],
    name='Assets'
)
liabilities = go.Bar(
    x=NKE_BS.index,
    y=NKE_BS["Total Liabilities"],
    name='Liabilities'
)

shareholder_equity = go.Scatter(
    x=NKE_BS.index,
    y=NKE_BS["Total Equity"],
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
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    asset_bar = go.Bar(
        x=NKE_BS.index,
        y=NKE_BS[ col ],
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
        x=NKE_BS.index,
        y=NKE_BS[ col ],
        name=col
    )    
    liability_data.append(liability_bar)
    
layout_liabilitys = go.Layout(
    barmode='stack'
)

fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)
py.iplot(fig_bs_liabilitys, filename='Total liabilities Breakdown')
NKE_PL[["Revenue"]].plot()
NKE_BS["working capital"] = NKE_BS["Total Current Assets"] - NKE_BS["Total Current Liabilities"]
NKE_BS[["working capital"]].plot()
NKE_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Using Chart Studio in Plotly 

PR_data = []
columns = '''
Accounts & Notes Receivable
Payables & Accruals
'''

for col in columns.strip().split("\n"):
    PR_Scatter = go.Scatter(
        x=NKE_BS.index,
        y=NKE_BS[ col ],
        name=col
    )    
    PR_data.append(PR_Scatter)
    
layout_PR = go.Layout(
    barmode='stack'
)

fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)
py.iplot(fig_bs_PR, filename='Accounts & Notes Receivable vs Payables & Accruals')
NKE_BS["Inventories"].plot()
NKE_BS[ ["Property, Plant & Equipment, Net",  "Other Long Term Assets"] ].plot()
NKE_BS
# Using Plotly

AAA_data = []
columns = '''
Property, Plant & Equipment, Net
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    AAA_bar = go.Bar(
        x=NKE_BS.index,
        y=NKE_BS[ col ],
        name=col
    )    
    AAA_data.append(AAA_bar)
    
layout_AAA = go.Layout(
    barmode='stack'
)

fig_bs_AAA = go.Figure(data=AAA_data, layout=layout_AAA)
py.iplot(fig_bs_AAA, filename='Total Long Term Assets')
equity_columns = '''
Preferred Equity
Share Capital & Additional Paid-In Capital
Retained Earnings
Other Equity
Equity Before Minority Interest
'''

equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
NKE_BS[ equity_columns ].plot()
# Using Plotly

equity_data = []
columns = '''
Preferred Equity
Share Capital & Additional Paid-In Capital
Retained Earnings
Other Equity
Equity Before Minority Interest
'''


for col in columns.strip().split("\n"):
    equity_Scatter = go.Scatter(
        x=NKE_BS.index,
        y=NKE_BS[ col ],
        name=col
    )    
    equity_data.append(equity_Scatter)
    
layout_equity = go.Layout(
    barmode='stack'
)

fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)
py.iplot(fig_bs_equity, filename='Total Equity')
NKE_BS["book value"] = NKE_BS["Total Assets"] - NKE_BS["Total Liabilities"]
NKE_BS["book value"].plot()
NKE_BS[["book value"]]
# Current Ratio

#current ratio = current asset / current liabilties
NKE_BS["current ratio"] = NKE_BS["Total Current Assets"] / NKE_BS["Total Current Liabilities"]
NKE_BS["current ratio"].plot()
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  
#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate
#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp

PE_RATIO = 35.26 # (Past Five Year's Data)FROM Ychart WEBSITE: https://ycharts.com/companies/NKE/pe_ratio 

# FROM YAHOO WEBSITE: https://finance.yahoo.com/quote/nke/analysis/
GROWTH_RATE = 0.1668 

PEG_ratio = PE_RATIO / (GROWTH_RATE*100)

print("NKE's PEG Ratio is", PEG_ratio)
NKE_PL
NKE_CF
#Additional Calculation 1 - CAGR
#Step 1 - EPS 
#EPS=(Net Income-Dividend Paid) / Common Shares Outstanding
Common_Shares_Outstanding = 1577000000 #From SIMFIN Website:https://simfin.com/data/companies/101814
Net_Income = NKE_PL["Net Income Available to Common Shareholders"]
EPS = (Net_Income / Common_Shares_Outstanding)*1000000
EPS
#Additional Calculation - CAGR 
#Step 1.1 - EPS Annual Compounded Growth Rate
#EPS Annual Compounded Growth Rate = (FV/PV)^(1/n) -1
FV = 1.225745
PV = 0.942930
n = 9
CAGR = ((FV/PV)**(1/n)) - 1
CAGR
#Additional Calculation 2 - Estimate EPS 10 Years from now
#FV = PV * (1+g)^n
PV =  1.225745
g = 0.029574678670879262
n = 10
Estimate_EPS = PV * (1+g)**n
Estimate_EPS
#Additional Calculation 3 - Current Target Buy Price
#Step 1.1-Estimate stock price 10 years from now
#Current_Target_Buy_Price = (Estimate_EPS * Avg_PE_Ratio)/(1+Discount_Rate)^10
Estimate_EPS = 1.6405091627187776
Avg_PE_Ratio = 35.26 #From https://ycharts.com/companies/NKE/pe_ratio
Discount_Rate = 0.09 #This is the industry average WACC from https://finbox.com/NYSE:NKE/models/wacc
Current_Target_Buy_Price = (Estimate_EPS * Avg_PE_Ratio)/(1+Discount_Rate)**10
Current_Target_Buy_Price
#Since the orginal excel sheet don't have "interest expense","Fix cost" I found a more detailed financial report in SIMFIN.
_2, NKE_PL2, NKE_BS2, NKE_CF2 = excel_to_df("/kaggle/input/nikes-financial-statement-with-2-version/SimFin-data_o.xlsx")
##Additional Calculation 4.1 - Marginal of Safety 
#Margin_of_Safety = (Net Sales - Break_Even_Point)/Net Sales
  #Break_Even_Point = FC/Contribution_Margin
  #Contribution_Margin = (Net_Sales - VC)/Net Sales = Gross_Margin / Net Sales

NKE_PL2["Fix_Cost"] = NKE_PL2["Demand creation expense"] + NKE_PL2["Operating overhead expense"] 
NKE_PL2["Variable_Cost"] = NKE_PL2["Cost of sales"]
Contribution_Margin = NKE_PL2["Gross Profit"]/NKE_PL2["Revenues"]
Break_Even_Point = NKE_PL2["Fix_Cost"]/Contribution_Margin
Margin_of_Safety = (NKE_PL2["Revenues"]-Break_Even_Point)/NKE_PL2["Revenues"]
#I will choose the average margin of safety to be the marginal of safety
Margin_of_Safety.mean()
#Additional Calculation 4.2 - Target Buy Price (add margin of safety - 30% off the target buy price)
#Target_Buy_Price = Current_Target_Buy_Price - (Current_Target_Buy_Price * 0.3)
Current_Target_Buy_Price = 24.43407985781073
Target_Buy_Price = Current_Target_Buy_Price - (Current_Target_Buy_Price * 0.3)
Target_Buy_Price
Margin_of_Safety
Margin_of_Safety.plot()
#Additional Calculation 5 - Debt to Equity Ratio
#DE_Ratio = Total Liability /  Total Equity
DE_Ratio=NKE_BS["Total Liabilities"]/ NKE_BS["Total Equity"]
DE_Ratio

NKE_BS["D/E"]=DE_Ratio#In order to drew a plot picture of "Interest_Coverage_Ratio", now I will add Interest_Coverage_Ratio into the dataset
NKE_BS["D/E"].plot()
NKE_PL2
#Additional Calculation 6 - Interest Coverage Ratio
#Interest_Coverage_Ratio = EBIT / Interest Expense
#EBIT = EBT + interest expense
EBIT = NKE_PL2["Income before income taxes"]-NKE_PL2["Interest expense (income), net"]
Interest_Coverage_Ratio=EBIT/NKE_PL2["Interest expense (income), net"]
Interest_Coverage_Ratio
Interest_Coverage_Ratio.abs()
NKE_PL2["ICR"]=Interest_Coverage_Ratio
NKE_PL2["EBIT"]=EBIT
#In order to drew a plot picture of "Interest_Coverage_Ratio", now I will add Interest_Coverage_Ratio into the dataset
ICR_columns = '''
EBIT
Interest expense (income), net
ICR
'''
ICR_columns = [ x for x in ICR_columns.strip().split("\n")]
NKE_PL2[ ICR_columns ].plot()
#End of Value Investing Stock Analysis Template