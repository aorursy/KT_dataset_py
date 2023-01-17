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

!pip install plotly==4.4.1

!pip install chart_studio

!pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe
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

_, Coty_PL, Coty_BS, Coty_CF = excel_to_df("/kaggle/input/COTY-Original.xlsx")
Coty_BS
del(Coty_BS["Assets"])
Coty_BS
# step one: Anual Compounded Growth Rate

# Get 10 years EPS from https://www.macrotrends.net/stocks/charts/COTY/coty/eps-earnings-per-share-diluted



# Coty Inc. Annual EPS from 2010 to 2019

EPS = [0.22, 0.18, -0.87, 0.42, -0.26, 0.64, 0.44, -0.66, -0.23, -5.04]



# Since in 2010, EPS is positive, while in 2019 EPS is negative, we cannot derive a CAGR.

# But we can see the simple rate of growth --- simpleGR



simpleGR = EPS[9]-EPS[0]/EPS[0]

print(simpleGR)

# step two: Estimate EL's EPS in 2029- EPS2029



EPS2029 = -EPS[9]*(1+simpleGR)

print (EPS2029)
# step three: calculate PE ratio.

# get annual average stock prices from - https://www.macrotrends.net/stocks/charts/COTY/coty/stock-price-history



# Since Coty has just went public in 2013, but from the rest US based peers, Coty is one of the most famous one.

# Coty's market Cao ranks the second in its industry in the US.



# Here are the stock pririces from 2013 to 2019

stockPrice = [16.2480,16.7094, 25.9594, 25.1655, 18.1462, 14.2642, 10.8865]

PE = []

for i in range(7):

    perYR_PE = stockPrice[i]/EPS[i]

    PE.append(perYR_PE)

from numpy import * 

avrgPE = mean(PE)



# estimated stock Price 10 years from now

COTYprice2029 = EPS2029*avrgPE

print(COTYprice2029)



    
# step four: get desired return rate. Desired return rate is considered as WACC. 

# get WACC from website. https://finbox.com/NYSE:COTY/models/wacc



# the lowest cost of equity shown on the website is 13.7%, and the highest cost of equity is 28.8%

# choose the average WACC, which is 8.5%



targetPrice = COTYprice2029 / (1+0.075)**10

print(targetPrice)

# margin of safety 100%

# Since no stock price will be negative, I will not invest Coty security.
# If the shareholer equity has the same trend as PB ratio, then the answer is yes.

# Book value; BV = Total Assets - Intangible assets - Liabilities - Preferred Stock Value

# No preferred stock can be found in the Coty xlsx



Coty_BS["book value"] = Coty_BS["Total assets"] - Coty_BS["Total liabilities"] - Coty_BS["Goodwill"] - Coty_BS["Other intangible assets, net"] - Coty_BS["Preferred stock"]

stockPrice = [16.2480,16.7094, 25.9594, 25.1655, 18.1462, 14.2642, 10.8865]



# Total outstanding shares of Class A Common Stock,in millions. Data are grabbed from SEC. https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001024305&type=10-k&dateb=&owner=exclude&count=40

StockNumber = [81.5,90.3,98.8,74.0,748.7,750.8,754.2]



PB = []

for i in range(7):

    perYR_PB = stockPrice[i]/(Coty_BS["book value"][i+1]/ StockNumber [i])

    PB.append(perYR_PB)

from numpy import *

avrgPB = mean(PB)
Coty_BS["book value"]
print(avrgPB)

print(PB[6])

print(PB)
from pandas.core.frame import DataFrame

pb = {"Price to BV": PB}

PtB = DataFrame(pb)

PtB.plot()
Coty_BS ["PB"] = PtB

BV_stuff = '''

PB

Total Coty Inc. stockholders' equity

'''

BV_stuff_column = [ x for x in BV_stuff.strip().split("\n") ]



BV_stuff_column
# try to put shareholer equity and PB ratio into one graph to see whether shareholer equity reflected in a low PB. 

Coty_BS[ BV_stuff_column ].plot()
# goodwill reflects a good customer relation and solid client base. To plot the goodwill changes from the past 7 years to see weather Coty has a substantial moat

Coty_BS[["Goodwill"]].plot()
import chart_studio

chart_studio.tools.set_credentials_file(username='Jessie_Huang', api_key='pqE3m2ZGOj9Un75uvT5Y')



import chart_studio.plotly as py

import plotly.graph_objs as go

#from tk_library_py import combine_regexes
# my first pricing power proxy is denoted to the rank of market cap in the industry

market_cap = 8,649,000,000 #get market cap from https://www.macrotrends.net/stocks/charts/COTY/coty/market-cap
# my second criterion of pricing power is the difference between net revenues and net income (loss) since the diffetence is the total cost that Coty must pay.

Coty_PL["price paid"] = Coty_PL["Net revenues"] - Coty_PL["Net income (loss) attributable to Coty Inc."]
# price paid VS net revenues, which is higher? If it is the former, then no pricing power

price_paid = go.Line(

    x=Coty_PL.index,

    y=Coty_PL["price paid"],

    name='price paid'

)





Net_revenues = go.Bar(

    x=Coty_PL.index,

    y=Coty_PL["Net revenues"],

    name='Net revenues'

)



data_pricePaidvsNetrevenue = [price_paid, Net_revenues]



layout_pricePaidvsNetrevenue = go.Layout(

    title = "price paid VS net revenues"

)



fig_pl_pricePaidvsNetrevenue = go.Figure(data=data_pricePaidvsNetrevenue, layout=layout_pricePaidvsNetrevenue)

fig_pl_pricePaidvsNetrevenue.show()

# Refer to the trends of revenue and of operating expense. If both of them are upwards, then Coty may sell more products.

SP_data = []

SalePotential_stuff = '''

Selling, general and administrative expenses

Acquisition-related costs

Gain on sale of assets

Net revenues

'''



for i in SalePotential_stuff.strip().split("\n"):

    SP_Line = go.Line(

        x=Coty_PL.index,

        y=Coty_PL[ i ],

        name=i

    )    

    SP_data.append(SP_Line)

    

layout_SP = go.Layout(

    barmode='stack'

)



fig_pl_sp = go.Figure(data=SP_data, layout=layout_SP)

fig_pl_sp.show()



#py.iplot(fig_pl_sp, filename='Sale Potential')

# to Calculate how many profits per cost can make and compare to the gross profit.



# calculate the ability of "cost of sales" to generate profits.

Coty_PL["profit per cost"] = Coty_PL["Gross Profit"] / Coty_PL["Cost of sales"]



# calculate the ability of "operating expenses" to generate profits.

Coty_PL["Total operating expenses"] = Coty_PL["Gross Profit"] - Coty_PL["Operating Income"]



Coty_PL["profit per op_expense"] = Coty_PL["Gross Profit"] / Coty_PL["Total operating expenses"]



# show their abilities

Coty_PL[["profit per cost", "profit per op_expense"]]

#  visualize their abilities



profit_per_cost = go.Bar(

    x=Coty_PL.index,

    y=Coty_PL["profit per cost"],

    name='profit per cost'

)





pp_op_expense = go.Bar(

    x=Coty_PL.index,

    y=Coty_PL["profit per op_expense"],

    name='profit per op expense'

)



data_PPCvsPPO = [profit_per_cost, pp_op_expense]



layout_PPCvsPPO = go.Layout(

    title = "profit per cost VS profit per op expense"

)



fig_pl_PPCvsPPO = go.Figure(data=data_PPCvsPPO, layout=layout_PPCvsPPO)

fig_pl_PPCvsPPO.show()

#py.iplot(fig_pl_PPCvsPPO, filename='profit per cost VS profit per op_expense')
# draw the trend of gross profit

profit_per_cost = go.Bar(

    x=Coty_PL.index,

    y=Coty_PL["profit per cost"],

    name='profit per cost'

)





pp_op_expense = go.Bar(

    x=Coty_PL.index,

    y=Coty_PL["profit per op_expense"],

    name='profit per op expense'

)



grossProfit = go.Scatter(

    x=Coty_PL.index,

    y=Coty_PL["Gross Profit"],

    name='Gross Profit'

)

data_gp = [grossProfit,profit_per_cost,pp_op_expense]

layout_gp = go.Layout(

    title = "Gross Profit Trends VS profits per expense"

)



fig_pl_gp = go.Figure(data=data_gp, layout=layout_gp)

fig_pl_gp.show()



#py.iplot(fig_pl_gp, filename='grossProfit')
# standardize the  gross profits to compare trends

st = []

for i in range(8):

    ST = Coty_PL["Gross Profit"][i]/(Coty_PL["Gross Profit"].max() - Coty_PL["Gross Profit"].min())

    st.append(ST)

standardized = {"st_gf": st}

ST_grossprofit = DataFrame(standardized, index = ["FY '12","FY '13","FY '14","FY '15","FY '16","FY '17","FY '18","FY '19"])



ST_grossprofit
# operating expenses stand for the overheads. Let us use elasticity to see the capacity of Coty to reduce its expense.

# the denominator should be the change ratio of operating expenses

# the numerator could be the change ratio of Coty's market Cap



denominator = []

for d in range(7):

    percentage_OF_OE = (Coty_PL["Total operating expenses"][d+1]-Coty_PL["Total operating expenses"][d])/Coty_PL["Total operating expenses"][d]

    denominator.append(percentage_OF_OE)



#stockPrice = [16.2480,16.7094, 25.9594, 25.1655, 18.1462, 14.2642, 10.8865]

#StockNumber = [81.5,90.3,98.8,74.0,748.7,750.8,754.2]

MarketCap = []

for m in range(7):

    mktCp = stockPrice[m] * StockNumber[m]

    MarketCap.append(mktCp)



numerator = []

for n in range(6):

    percentage_OF_MC = (MarketCap[n+1] - MarketCap[n]) / MarketCap[n]

    numerator.append(percentage_OF_MC)

    

elasticity = []

for e in range(6):

    E = denominator[e]/numerator[e]

    elasticity.append(E)

    

from pandas.core.frame import DataFrame

el = {"Elstcity": elasticity}

Elas = DataFrame(el)



Elas

    

    

    
# factors like goodwill, inventory and cost of revenue are crucial

# find the most important factor

moreSell = []

moreSell_stuff1 = '''

Goodwill

Inventories

'''

moreSell_stuff2 = '''

Cost of sales

Net revenues

'''

for col in moreSell_stuff1.strip().split("\n"):

    moreSell1_bar = go.Bar(

        x=Coty_BS.index,

        y=Coty_BS[ col ],

        name=col

    )    

    moreSell.append(moreSell1_bar)



for clm in moreSell_stuff2.strip().split("\n"):

    moreSell2_bar = go.Bar(

        x=Coty_PL.index,

        y=Coty_PL[ clm ],

        name=clm

    )    

    moreSell.append(moreSell2_bar)

    

layout_moreSell = go.Layout(

    title = "goodwill, inventory and cost of revenue, net revenues"

)



fig_sheet_moreSell = go.Figure(data=moreSell, layout=layout_moreSell)

fig_sheet_moreSell.show()
# According to investment activities, to see Coty's investment plan on different projects

investment_data = []

columns = '''

Capital expenditures

Payments for business combinations, net of cash acquired

Additions of goodwill

Proceeds from sale of assets

Cash acquired from business combination

Payments related to loss on foreign currency contracts

'''





for col in columns.strip().split("\n"):

    investment_bar = go.Bar(

        x=Coty_CF.index,

        y=Coty_CF[ col ],

        name=col

    )    

    investment_data.append(investment_bar)

    

layout_investment = go.Layout(

)



fig_cf_investment = go.Figure(data=investment_data, layout=layout_investment)

fig_cf_investment.show()

#py.iplot(fig_cf_investment, filename='investment')
Coty_CF["Net cash used in investing activities"].plot()
#stock Buyback

Coty_BS["Buyback"] = - Coty_BS["Treasury stock--at cost"]

Buyback = go.Scatter(

    x=Coty_BS.index,

    y=Coty_BS["Buyback"],

    name='Buyback'

)



data_Buyback = Buyback

layout_Buyback = go.Layout(

    title = "Coty stock buyback"

)



fig_bs_Buyback = go.Figure(data=data_Buyback, layout=layout_Buyback)

fig_bs_Buyback.show()

#py.iplot(fig_bs_Buyback, filename='Buyback')

# Book value can be seen as the liquidtion value

Coty_BS["book value"] = Coty_BS["Total assets"] - Coty_BS["Total liabilities"] - Coty_BS["Goodwill"] - Coty_BS["Other intangible assets, net"] - Coty_BS["Preferred stock"]

Coty_BS[["book value"]]
Coty_BS[["book value"]].plot()
# calculate interest coverage ratio

interest_stuff = '''

Operating Income

INCOME (LOSS) BEFORE INCOME TAXES

'''



interest_columns = [ x for x in interest_stuff.strip().split("\n") ]
# interest expense dataframe equals to the difference between "Operating Income" and "INCOME (LOSS) BEFORE INCOME TAXES", but we need to add back non-operating income(loss)



Coty_PL["interest expense"] = Coty_PL[interest_columns[0]] - Coty_PL[interest_columns[1]] + Coty_PL["Other expense, net"]
# EBIT dataframe 

# EBIT can be seen as the operating income. But for Coty Inc, non-operating income--"Other expense, net", is listed in P&L, which does influence the EBIT.

Coty_PL["EBIT"] = Coty_PL["Operating Income"] - Coty_PL["Other expense, net"]
Coty_PL[["interest expense","EBIT" ]]
# interest coverage ratio Dataframe

Coty_PL["ICR"] = Coty_PL["EBIT"] / Coty_PL["interest expense"]
Coty_PL["ICR"]
# use matplotlib to graph interest rate

%matplotlib inline

Coty_PL[["ICR"]].plot()
Coty_BS["DtE"] = Coty_BS["Total liabilities"] / Coty_BS["Total equity"]



DtE = go.Scatter(

    x=Coty_BS.index,

    y=Coty_BS["DtE"],

    name='DtE'

)



data_DtE = DtE

layout_DtE = go.Layout(

    title = "Coty Inc Debt to Equity Ratio"

)



fig_bs_DtE = go.Figure(data=data_DtE, layout=layout_DtE)

fig_bs_DtE.show()