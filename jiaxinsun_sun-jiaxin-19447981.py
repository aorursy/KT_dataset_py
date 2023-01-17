# Do the below in kaggle

#!pip install plotly==4.4.1

#!pip install chart_studio

#!pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe
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
# Add 'tk_library.py' file given by your tutor, as a utility script under 'File'

# Look for it under usr/bin on the right drawer



# import excel_to_df function

import os

#import tk_library_py

#from tk_library_py import excel_to_df
# Show the files and their pathnames

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Add your simfin-data.xlsx using the '+ Add Data' top right button

_, DKS_PL, DKS_BS, DKS_CF = excel_to_df("/kaggle/input/dickssporting/SimFin-data (1).xlsx")
DKS_BS
del(DKS_BS["Assets"])

DKS_BS
DKS_BS["_Total Current Assets"] = DKS_BS["Cash, Cash Equivalents & Short Term Investments"] + DKS_BS["Inventories"] + DKS_BS["Other Short Term Assets"]
DKS_BS[["_Total Current Assets", "Total Current Assets"]]
DKS_BS["_NonCurrent Assets"] = DKS_BS["Property, Plant & Equipment, Net"] + DKS_BS["Long Term Investments & Receivables"] + DKS_BS["Other Long Term Assets"]
DKS_BS["_Total Assets"] = DKS_BS["_NonCurrent Assets"] + DKS_BS["_Total Current Assets"] 
DKS_BS["_Total Liabilities"] = DKS_BS["Total Current Liabilities"] + DKS_BS["Total Noncurrent Liabilities"]
DKS_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline

DKS_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''



asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
DKS_BS[ asset_columns ].plot()
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

    x=DKS_BS.index,

    y=DKS_BS["Total Assets"],

    name='Assets'

)

liabilities = go.Bar(

    x=DKS_BS.index,

    y=DKS_BS["Total Liabilities"],

    name='Liabilities'

)



shareholder_equity = go.Scatter(

    x=DKS_BS.index,

    y=DKS_BS["Total Equity"],

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

Long Term Investments & Receivables

Other Long Term Assets

'''





for col in columns.strip().split("\n"):

    asset_bar = go.Bar(

        x=DKS_BS.index,

        y=DKS_BS[ col ],

        name=col

    )    

    asset_data.append(asset_bar)

    

layout_assets = go.Layout(

    barmode='stack'

)



fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)

fig_bs.show()
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

        x=DKS_BS.index,

        y=DKS_BS[ col ],

        name=col

    )    

    liability_data.append(liability_bar)

    

layout_liabilitys = go.Layout(

    barmode='stack'

)



fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)

fig_bs.show()
DKS_PL[["Revenue"]].plot()
DKS_BS["working capital"] = DKS_BS["Total Current Assets"] - DKS_BS["Total Current Liabilities"]
DKS_BS[["working capital"]].plot()
DKS_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Using Chart Studio in Plotly 



PR_data = []

columns = '''

Accounts & Notes Receivable

Payables & Accruals

'''



for col in columns.strip().split("\n"):

    PR_Scatter = go.Scatter(

        x=DKS_BS.index,

        y=DKS_BS[ col ],

        name=col

    )    

    PR_data.append(PR_Scatter)

    

layout_PR = go.Layout(

    barmode='stack'

)



fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)

fig_bs.show()
DKS_BS["Inventories"].plot()
DKS_BS[ ["Property, Plant & Equipment, Net", "Long Term Investments & Receivables", "Other Long Term Assets"] ].plot()
DKS_BS

# Using Plotly



AAA_data = []

columns = '''

Property, Plant & Equipment, Net

Long Term Investments & Receivables

Other Long Term Assets

'''





for col in columns.strip().split("\n"):

    AAA_bar = go.Bar(

        x=DKS_BS.index,

        y=DKS_BS[ col ],

        name=col

    )    

    AAA_data.append(AAA_bar)

    

layout_AAA = go.Layout(

    barmode='stack'

)



fig_bs_AAA = go.Figure(data=AAA_data, layout=layout_AAA)

fig_bs.show()
equity_columns = '''

Share Capital & Additional Paid-In Capital

Treasury Stock

Retained Earnings

Other Equity

Equity Before Minority Interest

'''



equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
DKS_BS[ equity_columns ].plot()
# Using Plotly



equity_data = []

columns = '''

Share Capital & Additional Paid-In Capital

Treasury Stock

Retained Earnings

Other Equity

Equity Before Minority Interest

'''





for col in columns.strip().split("\n"):

    equity_Scatter = go.Scatter(

        x=DKS_BS.index,

        y=DKS_BS[ col ],

        name=col

    )    

    equity_data.append(equity_Scatter)

    

layout_equity = go.Layout(

    barmode='stack'

)



fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)

fig_bs.show()
DKS_BS["book value"] = DKS_BS["Total Assets"] - DKS_BS["Total Liabilities"]
DKS_BS["book value"].plot()

DKS_BS[["book value"]]
DKS_BS["current ratio"] = DKS_BS["Total Current Assets"] / DKS_BS["Total Current Liabilities"]
DKS_BS["current ratio"].plot()
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  

#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate

#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp



PE_RATIO = 14.07 # (use past 5 years average PE ratio)FROM Ycharts WEBSITE: https://ycharts.com/companies/DKS/pe_ratio



# FROM YAHOO WEBSITE: https://finance.yahoo.com/quote/DKS/analysis?p=DKS

GROWTH_RATE = 0.066 

PEG_ratio = PE_RATIO / (GROWTH_RATE*100)



print("DICKS SPORTING Corp's PEG Ratio is", PEG_ratio)
DKS_PL

DKS_CF
#Additional Calculation 1 - CAGR

#Step 1 - EPS 

#EPS=(Net Income-Dividend Paid) / Common Shares Outstanding

Common_Shares_Outstanding = 91246261 #From SIMFIN Website:https://simfin.com/data/companies/179423

Net_Income=DKS_PL["Net Income Available to Common Shareholders"]

EPS = (Net_Income / Common_Shares_Outstanding)*1000000

EPS
#Additional Calculation - CAGR

#Step 1.1 - EPS Annual Compounded Growth Rate

#EPS Annual Compounded Growth Rate = (FV/PV)^(1/n) -1

FV = 3.505503

PV = 1.483447

n = 9

CAGR = ((FV/PV)**(1/n)) - 1

CAGR
#Additional Calculation 2 - Estimate EPS 10 Years from now

#FV = PV * (1+g)^n

PV = 3.505503

g = 0.10026573922468351

n = 10

Estimate_EPS = PV * (1+g)**n

Estimate_EPS
#Additional Calculation 3 - Current Target Buy Price

#Step 1.1-Estimate stock price 10 years from now

#Current_Target_Buy_Price = (Estimate_EPS * Avg_PE_Ratio)/(1+Discount_Rate)^10

Estimate_EPS = 9.114361322919072

Avg_PE_Ratio = 12.02 #From SIMFIN Website:https://simfin.com/data/companies/179423

Discount_Rate = 0.1050 #This is the industry average WACC from https://finbox.com/NYSE:DKS/models/wacc

Current_Target_Buy_Price = (Estimate_EPS * Avg_PE_Ratio)/(1+Discount_Rate)**10

Current_Target_Buy_Price
#Since the orginal excel sheet don't have "interest expense","Fix cost" I found a more detailed financial report in SIMFIN.

_2, DKS_PL2, DKS_BS2, DKS_CF2 = excel_to_df("/kaggle/input/more-detailed-data-for-dks/SimFin-data (2).xlsx")
##Additional Calculation 4.1 - Marginal of Safety 

#Margin_of_Safety = (Net Sales - Break_Even_Point)/Net Sales

  #Break_Even_Point = FC/Contribution_Margin

  #Contribution_Margin = (Net_Sales - VC)/Net Sales = Gross_Margin / Net Sales



DKS_PL2["Fix_Cost"] = DKS_PL2["Selling, general and administrative expenses"] + DKS_PL2["Business Combination, Integration Related Costs"] + DKS_PL2["Pre-opening expenses"]

DKS_PL2["Variable_Cost"] = DKS_PL2["Cost of goods sold, including occupancy and distribution costs"]

Contribution_Margin = DKS_PL2["Gross Profit"]/DKS_PL2["Net sales"]

Break_Even_Point = DKS_PL2["Fix_Cost"]/Contribution_Margin

Margin_of_Safety = (DKS_PL2["Net sales"]-Break_Even_Point)/DKS_PL2["Net sales"]

#I will choose the average margin of safety to be the marginal of safety

Margin_of_Safety.mean()
#Additional Calculation 4.2 - Target Buy Price (add margin of safety - 23% off the target buy price)

#Target_Buy_Price = Current_Target_Buy_Price - (Current_Target_Buy_Price * 0.23)

Current_Target_Buy_Price = 40.365276236482494

Target_Buy_Price = Current_Target_Buy_Price - (Current_Target_Buy_Price * 0.23)

Target_Buy_Price
Margin_of_Safety
Margin_of_Safety.plot()
#Additional Calculation 5 - Debt to Equity Ratio

#DE_Ratio = Total Liability /  Total Equity

DE_Ratio=DKS_BS["_Total Liabilities"]/ DKS_BS["Total Equity"]

DE_Ratio

DKS_BS["D/E"]=DE_Ratio#In order to drew a plot picture of "Interest_Coverage_Ratio", now I will add Interest_Coverage_Ratio into the dataset

DKS_BS["D/E"].plot()
DKS_PL2
#Additional Calculation 6 - Interest Coverage Ratio

#Interest_Coverage_Ratio = EBIT / Interest Expense

Interest_Coverage_Ratio=DKS_PL2["Income from operations"]/DKS_PL2["Interest expense"]

Interest_Coverage_Ratio

DKS_PL2["ICR"]=Interest_Coverage_Ratio#In order to drew a plot picture of "Interest_Coverage_Ratio", now I will add Interest_Coverage_Ratio into the dataset
ICR_columns = '''

Income from operations

Interest expense

ICR

'''

ICR_columns = [ x for x in ICR_columns.strip().split("\n")]

DKS_PL2[ ICR_columns ].plot()
#End of Value Investing Stock Analysis Template