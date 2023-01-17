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

#!pip install plotly==4.4.1

#!pip install chart_studio

#!pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe
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

_, Estee_PL, Estee_BS, Estee_CF = excel_to_df("/kaggle/input/estlaunder/ELoriginal.xlsx")
Estee_BS
del(Estee_BS["Assets"])
# step one: Anual Compounded Growth Rate - grwth_EPS

# Get 10 years EPS from  https://www.macrotrends.net/stocks/charts/EL/estee-lauder/eps-earnings-per-share-diluted

# EStess Lauder Annual EPS from 2010 to 2019

EPS = [1.19, 1.74, 2.16, 2.58, 3.06, 2.82, 2.96, 3.35, 2.95, 4.82]

# Calculate the Anual Compounded Growth Rate - grwth_EPS

grwth_EPS = (EPS[9]/EPS[0])**0.1-1

print(grwth_EPS)



# step two: Estimate EL's EPS in 2029- EPS2029

EPS2029 = EPS[9]*(1+grwth_EPS)**10

print (EPS2029)
# step three: calculate PE ratio.

# get annual average stock prices from - https://www.macrotrends.net/stocks/charts/EL/estee-lauder/stock-price-history

stockPrice = [24.27,40.22, 56.215, 61.01, 73.71, 75.59, 77.33, 87.21, 127.59, 131.94,]

PE = []

for i in range(10):

    perYR_PE = stockPrice[i]/EPS[i]

    PE.append(perYR_PE)

from numpy import *

avrgPE = mean(PE)



# estimated stock Price 10 years from now

ELprice2029 = EPS2029*avrgPE

print(ELprice2029)



    
# step four: get desired return rate. Desired return rate is considered as WACC. 

# get WACC from website. https://finbox.com/NYSE:EL/models/wacc

# choose the average WACC, which is 7.5%

targetPrice = ELprice2029 / (1+0.075)**10

print(targetPrice)

# margin of safety 15%

adjustedTargetPrice = targetPrice*(1-0.15)

print(adjustedTargetPrice)
# calculate interest coverage ratio

interest_stuff = '''

Interest expense

Interest expense on debt extinguishment

Other components of net periodic benefit cost

'''



interest_columns = [ x for x in interest_stuff.strip().split("\n") ]
# interest expense dataframe 

Estee_PL["interest expense"] = Estee_PL[interest_columns[0]] + Estee_PL[interest_columns[1]] + Estee_PL[interest_columns[2]]
# EBIT dataframe 

income_added = '''

Interest income and investment income, net

Other income+

'''

income_add_columns = [ x for x in interest_stuff.strip().split("\n") ]



Estee_PL["EBIT"] = Estee_PL["Operating Income"] + Estee_PL[income_add_columns[0]] + Estee_PL[income_add_columns[1]]
Estee_PL[["interest expense","EBIT" ]]
# interest coverage ratio Dataframe

Estee_PL["ICR"] = Estee_PL["EBIT"] / Estee_PL["interest expense"]
# use matplotlib to graph interest rate

%matplotlib inline

Estee_PL[["ICR"]].plot()
# Use Ploty to gragh "Debt to equity ratio"



import chart_studio

chart_studio.tools.set_credentials_file(username='Jessie_Huang', api_key='pqE3m2ZGOj9Un75uvT5Y')



import chart_studio.plotly as py

import plotly.graph_objs as go

#from tk_library_py import combine_regexes





Estee_BS["DtE"] = Estee_BS["Total Liabilities"] / Estee_BS["Total equity"]



DtE = go.Scatter(

    x=Estee_BS.index,

    y=Estee_BS["DtE"],

    name='DtE'

)



data_DtE = DtE

layout_DtE = go.Layout(

)



fig_bs_DtE = go.Figure(data=data_DtE, layout=layout_DtE)

fig_bs_DtE.show()

#py.iplot(fig_bs_DtE, filename='Total Assets and Liabilities')
# Book value; BV = Total Assets - Intangible assets - Liabilities - Preferred Stock Value

# No preferred stock can be found in the EL xlsx



Estee_BS["book value"] = Estee_BS["Total assets"] - Estee_BS["Total Liabilities"] - Estee_BS["Goodwill"] - Estee_BS["Other intangible assets, net"] 

stockPrice = [24.27,40.22, 56.215, 61.01, 73.71, 75.59, 77.33, 87.21, 127.59, 131.94,]

PB = []

for i in range(10):

    perYR_PB = stockPrice[i]/Estee_BS["book value"][i]

    PB.append(perYR_PB)

from numpy import *

avrgPB = mean(PB)
print(avrgPB)

print(PB[9])

print(PB)
from pandas.core.frame import DataFrame

pb = {"Price to BV": PB}

PtB = DataFrame(pb)

PtB.plot()
#margin of safety

Estee_PL[["Goodwill impairment"]].plot()
# pricing power

# my pricing power proxy is denoted to the rank of market cap in the industry

market_cap = 75,000,000,000 #get market cap from https://www.macrotrends.net/stocks/charts/EL/estee-lauder/market-cap
# sale potential

# Refer to the trends of Revenue and of operating expense

SP_data = []

SalePotential_stuff = '''

Selling, general and administrative

Restructuring and other charges

Goodwill impairment

Impairment of other intangible assets

Net sales

'''



for i in SalePotential_stuff.strip().split("\n"):

    SP_Line = go.Line(

        x=Estee_PL.index,

        y=Estee_PL[ i ],

        name=i

    )    

    SP_data.append(SP_Line)

    

layout_SP = go.Layout(

    barmode='stack'

)



fig_pl_sp = go.Figure(data=SP_data, layout=layout_SP)

fig_pl_sp.show()

#py.iplot(fig_pl_sp, filename='Sale Potential')

# margin pressure

Estee_PL["profit per cost"] = Estee_PL["Gross Profit"] / Estee_PL["Cost of sales"]

Estee_PL["profit per op_expense"] = Estee_PL["Gross Profit"] / Estee_PL["Total operating expenses"]



Estee_PL[["profit per cost", "profit per op_expense"]]

profit_per_cost = go.Scatter(

    x=Estee_PL.index,

    y=Estee_PL["profit per cost"],

    name='profit per cost'

)



data_PPC = profit_per_cost

layout_PPC = go.Layout(

)



fig_pl_PPC = go.Figure(data=data_PPC, layout=layout_PPC)

fig_pl_PPC.show()

#py.iplot(fig_pl_PPC, filename='profit per cost')
grossProfit = go.Scatter(

    x=Estee_PL.index,

    y=Estee_PL["Gross Profit"],

    name='Gross Profit'

)

data_gp = grossProfit

layout_gp = go.Layout(

)



fig_pl_gp = go.Figure(data=data_gp, layout=layout_gp)

fig_pl_gp.show()

#py.iplot(fig_pl_gp, filename='grossProfit')
# margin pressure



profit_per_cost = go.Bar(

    x=Estee_PL.index,

    y=Estee_PL["profit per cost"],

    name='profit per cost'

)



pp_op_expense = go.Bar(

    x=Estee_PL.index,

    y=Estee_PL["profit per op_expense"],

    name='profit per op expense'

)



data_PPCvsPPO = [profit_per_cost, pp_op_expense]



layout_PPCvsPPO = go.Layout(

    title = "profit per cost VS profit per op expense"

)



fig_pl_PPCvsPPO = go.Figure(data=data_PPCvsPPO, layout=layout_PPCvsPPO)

fig_pl_PPCvsPPO.show()

#py.iplot(fig_pl_PPCvsPPO, filename='profit per cost VS profit per op_expense')
list = []

for i in range(9):

    ppcGrowth = (Estee_PL["profit per cost"][i+1]- Estee_PL["profit per cost"][i]) / Estee_PL["profit per cost"][i]

    list.append(ppcGrowth)

from pandas.core.frame import DataFrame

g = {"ppcGrowth": list}

Growth = DataFrame(g)



Growth.plot()
# overheads

CFinoverheads = go.Bar(

    x=Estee_CF.index,

    y=Estee_CF["Goodwill and other intangible asset impairments"],

    name='CFinoverheads'

)



data_CFinoverheads = CFinoverheads

layout_CFinoverheads = go.Layout(

)



fig_cf_CFinoverheads = go.Figure(data=data_CFinoverheads, layout=layout_CFinoverheads)

fig_cf_CFinoverheads.show()

#py.iplot(fig_cf_CFinoverheads, filename='CFinoverheads')
# Investment plans

investment_data = []

columns = '''

Capital expenditures

Payments for acquired businesses, net of cash acquired

Proceeds from the disposition of investments

Purchases of investments

'''





for col in columns.strip().split("\n"):

    investment_bar = go.Bar(

        x=Estee_CF.index,

        y=Estee_CF[ col ],

        name=col

    )    

    investment_data.append(investment_bar)

    

layout_investment = go.Layout(

)



fig_cf_investment = go.Figure(data=investment_data, layout=layout_investment)

fig_cf_investment.show()

#py.iplot(fig_cf_investment, filename='investment')
Estee_CF["Net cash flows used for investing activities"].plot()
#stock Buyback

Estee_BS["Buyback"] = - Estee_BS["Less: Treasury stock"]

Buyback = go.Scatter(

    x=Estee_BS.index,

    y=Estee_BS["Buyback"],

    name='Buyback'

)



data_Buyback = Buyback

layout_Buyback = go.Layout(

)



fig_bs_Buyback = go.Figure(data=data_Buyback, layout=layout_Buyback)

fig_bs_Buyback.show()

#py.iplot(fig_bs_Buyback, filename='Buyback')

# Book value; BV = Total Assets - Intangible assets - Liabilities - Preferred Stock Value

# No preferred stock can be found in the EL xlsx



Estee_BS["book value"] = Estee_BS["Total assets"] - Estee_BS["Total Liabilities"] - Estee_BS["Goodwill"] - Estee_BS["Other intangible assets, net"] 
Estee_BS[["book value"]]
Estee_BS[["book value"]].plot()
#End of Value Investing Stock Analysis Template