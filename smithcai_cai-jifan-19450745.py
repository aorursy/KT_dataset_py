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
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#!pip install plotly==4.4.1

#!pip install chart_studio

#!pip install xlrd

import os

#import tk_library_py

#from tk_library_py import excel_to_df
for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_excel("/kaggle/input/google/SimFin-data (ALPHABET).xlsx")

data
title, google_PL, google_BS, google_CF = excel_to_df("/kaggle/input/google/SimFin-data (ALPHABET).xlsx")
google_BS
del(google_BS["Assets"])

google_BS
google_BS["Total Current Assets_"] = google_BS["Cash, Cash Equivalents & Short Term Investments"] + google_BS["Accounts & Notes Receivable"] + google_BS["Inventories"] + google_BS["Other Short Term Assets"]

google_BS[["Total Current Assets_", "Total Current Assets"]]
google_BS["Total NonCurrent Assets_"] = google_BS["Property, Plant & Equipment, Net"] + google_BS["Long Term Investments & Receivables"] + google_BS["Other Long Term Assets"]

google_BS["Total Assets_"] = google_BS["Total Current Assets_"] + google_BS["Total NonCurrent Assets_"]

google_BS[["Total Assets_","Total Assets"]]

google_BS["Total Liabilities_"] = google_BS["Total Current Liabilities"] + google_BS["Total Noncurrent Liabilities"]
google_BS[["Total Liabilities_", "Total Liabilities"]]
google_BS.loc[:, ['Total Assets', 'Total Liabilities', 'Total Equity']]
%matplotlib inline

google_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
google_BS.loc[:, ['Cash, Cash Equivalents & Short Term Investments', 'Accounts & Notes Receivable', 'Inventories','Other Short Term Assets']]
items = '''

Cash, Cash Equivalents & Short Term Investments.Accounts & Notes Receivable.Inventories.Other Short Term Assets

'''



asset_columns = [ i for i in items.strip().split(".") ]
asset_columns
google_BS[ asset_columns ].plot()
#!pip install chart_studio

import chart_studio

chart_studio.tools.set_credentials_file(username='smithcai', api_key='ayF0GBpENuRkssQdRQdo')
import chart_studio.plotly as py

import plotly.graph_objs as go

#from tk_library_py import combine_regexes
assets = go.Bar(

    x=google_BS.index,

    y=google_BS["Total Assets"],

    name='Assets'

)

liabilities = go.Bar(

    x=google_BS.index,

    y=google_BS["Total Liabilities"],

    name='Liabilities'

)



equity = go.Scatter(

    x=google_BS.index,

    y=google_BS["Total Equity"],

    name='Equity'

)



A = [assets, liabilities, equity]

B = go.Layout(

    barmode='stack'

)



C = go.Figure(data=A, layout=B)

py.plot(C, filename='Total Assets and Liabilities')

C.show()
asset_breakdown = []

breakdowns = '''

Cash, Cash Equivalents & Short Term Investments.Accounts & Notes Receivable.Inventories.Other Short Term Assets.Property, Plant & Equipment, Net.Long Term Investments & Receivables.Other Long Term Assets

'''





for bd in breakdowns.strip().split("."):

    asset_bar = go.Bar(

        x=google_BS.index,

        y=google_BS[ bd ],

        name=bd

    )    

    asset_breakdown.append(asset_bar)

A=asset_breakdown

    

layout_assets = go.Layout(

    barmode='stack'

)

B=layout_assets



C= go.Figure(data=A, layout=B)

py.plot(C, filename='Total Assets Breakdown')

C.show()
liability_breakdown = []

breakdowns = '''

Payables & Accruals.Short Term Debt.Other Short Term Liabilities.Long Term Debt.Other Long Term Liabilities

'''





for bd in breakdowns.strip().split("."):

    liability_bar = go.Bar(

        x=google_BS.index,

        y=google_BS[ bd ],

        name=bd

    )    

    liability_breakdown.append(liability_bar)

A=liability_breakdown

    

layout_liabilitys = go.Layout(

    barmode='stack'

)

B=layout_liabilitys



C= go.Figure(data=A, layout=B)

py.plot(C, filename='Total liabilities Breakdown')

C.show()
google_BS["working capital"] = google_BS["Total Current Assets"] - google_BS["Total Current Liabilities"]

google_BS[["working capital"]].plot()
google_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
PR_data = []

PR = '''

Accounts & Notes Receivable.Payables & Accruals

'''



for pr in PR.strip().split("."):

    PR_Scatter = go.Scatter(

        x=google_BS.index,

        y=google_BS[ pr ],

        name=pr

    )    

    PR_data.append(PR_Scatter)

A=PR_data

    

layout_PR = go.Layout(

    barmode='stack'

)

B=layout_PR



C = go.Figure(data=A, layout=B)

py.plot(C, filename='Accounts & Notes Receivable vs Payables & Accruals')

C.show()

google_BS[["Inventories"]].plot()
google_BS[ ["Property, Plant & Equipment, Net", "Long Term Investments & Receivables", "Other Long Term Assets"] ].plot()
LA_breakdown = []

breakdowns = '''

Property, Plant & Equipment, Net.Long Term Investments & Receivables.Other Long Term Assets

'''





for bd in breakdowns.strip().split("."):

    LA_bar = go.Bar(

        x=google_BS.index,

        y=google_BS[ bd ],

        name=bd

    )    

    LA_breakdown.append(LA_bar)

A=LA_breakdown

    

layout_LA = go.Layout(

    barmode='stack'

)

B=layout_LA



C= go.Figure(data=A, layout=B)

py.plot(C, filename='Total Long Term Assets')

C.show()
equity_breakdown = '''

Share Capital & Additional Paid-In Capital.Retained Earnings.Other Equity.Equity Before Minority Interest

'''



equity_breakdown = [ x for x in equity_breakdown.strip().split(".")]

google_BS[ equity_breakdown ].plot()
equity_breakdown = []

breakdowns = '''

Share Capital & Additional Paid-In Capital.Retained Earnings.Other Equity.Equity Before Minority Interest

'''





for bd in breakdowns.strip().split("."):

    equity_Scatter = go.Scatter(

        x=google_BS.index,

        y=google_BS[ bd ],

        name=bd

    )    

    equity_breakdown.append(equity_Scatter)

A=equity_breakdown

    

layout_equity = go.Layout(

    barmode='stack'

)

B=layout_equity



C= go.Figure(data=A, layout=B)

py.plot(C, filename='Total Equity')

C.show()
google_BS["book value"] = google_BS["Total Assets"] - google_BS["Total Liabilities"]

google_BS["book value"]

google_BS["book value"].plot()
google_BS["current ratio"] = google_BS["Total Current Assets"] / google_BS["Total Current Liabilities"]

print(google_BS["current ratio"])

google_BS["current ratio"].plot()
PE_Ratio = 63.99

# FROM SIMFIN WEBSITE: https://simfin.com/data/companies/61595

GROWTH_RATE =  0.1582

# FROM YAHOOFINANCE WEBSITE: https://finance.yahoo.com/quote/GOOG/analysis?p=GOOG

# Forcast over the five next years

PEG_ratio = PE_Ratio / (GROWTH_RATE*100)

print("Google Inc.'s PEG Ratio is", PEG_ratio)

#following are six additional calculations
#Earnings Per Share (EPS) Annual Compounded Growth Rate

#Formula and Calculation of CAGR: https://www.investopedia.com/terms/c/cagr.asp

#CAGR = (EB/BB)^1/n-1, where:EB=Ending balance, BB=Beginning balance, n=Number of years

#data from: https://www.macrotrends.net/stocks/charts/GOOG/alphabet/eps-earnings-per-share-diluted,https://www.macrotrends.net/stocks/charts/GOOGL/alphabet/eps-basic-net-earnings-per-share



import pandas as pd

EPS=pd.DataFrame({'12/30/2019': [49.59, 49.16],

                  '12/30/2018': [44.22, 43.70],

                  '12/30/2017': [18.27,18.00 ],

                  '12/30/2016': [28.32, 27.85],

                  '12/30/2015': [23.11,22.84 ],

                  '12/30/2014': [20.91,20.57 ],

                  '12/30/2013': [19.13,18.79 ],

                  '12/30/2012': [16.41,16.16 ],

                  '12/30/2011': [15.09,14.88 ],

                  '12/30/2010': [13.35,13.16 ]

                                              },

             index=['Reported EPS Basic','Reported EPS Diluted'])









EPS.loc[EPS['12/30/2019']==49.59]
EPS_ACGR_basic = ((49.59/13.35)**(1/9))-1

EPS_ACGR_basic
EPS.loc[EPS['12/30/2019']==49.16]
EPS_ACGR_diluted = ((49.16/13.16)**(1/9))-1

EPS_ACGR_diluted
print("Google Inc.'s basic and diluted EPS ACGR are",EPS_ACGR_basic,"and",EPS_ACGR_diluted,"respectively")
#EPS = a + b * t, where a = EPS @ t0, b = EPS growth, t=number of year

#Formula and Calculation of Estimate EPS: https://m.book118.com/html/2017/0902/131440384.shtm?from=singlemessage

#EPS_ACGR calculated above is not included in the belowing calculation because it comes from data of only two years

#instead average eps growth is used

data=EPS.T

data=data.iloc[::-1]

data['Basic growth(%)']= (data["Reported EPS Basic"]-data["Reported EPS Basic"].shift(1))/data["Reported EPS Basic"].shift(1)*100

data['Diluted growth(%)']= (data["Reported EPS Diluted"]-data["Reported EPS Diluted"].shift(1))/data["Reported EPS Diluted"].shift(1)*100

data
#a1=Reported EPS Basic@2019, a2=Reported EPS Diluted@2019, b1=average basic growth, b2=average diluted growth

a1 = data.iloc[9,0]

b1 = data['Basic growth(%)'].sum()/(100*9)

print(a1,b1)





a2 = data.iloc[9,1]

b2 = data['Diluted growth(%)'].sum()/(100*9)

print(a2,b2)



predicted_EPS=pd.DataFrame({'Expected Reported EPS Basic': [a1+b1*1,a1+b1*2,a1+b1*3,a1+b1*4,a1+b1*5,a1+b1*6,a1+b1*7,a1+b1*8,a1+b1*9,a1+b1*10,],

                        'Expected Reported EPS Diluted': [a2+b2*1,a2+b2*2,a2+b2*3,a2+b2*4,a2+b2*5,a2+b2*6,a2+b2*7,a2+b2*8,a2+b2*9,a2+b2*10,]},

                           index=range(2020,2030))





print(predicted_EPS)

predicted_EPS.plot()







EPS_10_YR=[]

ss = '''

Expected Reported EPS Basic/Expected Reported EPS Diluted

'''



for s in ss.strip().split("/"):

    EPS_10_YR_Scatter = go.Scatter(

        x=predicted_EPS.index,

        y=predicted_EPS[ s ],

        name=s)    



EPS_10_YR.append(EPS_10_YR_Scatter)

A=EPS_10_YR

    

layout_EPS_10_YR = go.Layout(

    barmode='stack')

B=layout_EPS_10_YR



C= go.Figure(data=A, layout=B)

py.plot(C, filename='EPS_10_YR')

C.show()
#Current Target Buy Price = EPS * expanded P/E, where expanded P/E = P/E * (1+130%), source: https://www.investors.com/how-to-invest/investors-corner/pe-expansion-can-help-set-a-price-target/

#P/E=share price/Net Income Available to Common Shareholders/Total weighted average shares outstanding=1,508.79M/34,343M/693M = 30.43, source: https://simfin.com/data/companies/18

#Alphabet(Google) Annual PE: FY'15:34.09, FY'16:28.47, FY'17:58.65, FY'18:23.91, FY'19:27.23, source: https://www.gurufocus.com/term/pe/NAS:GOOGL/PE-Ratio/Alphabet(Google)

#Alphabet(Google) Annual WACC: FY'15:7.51%, FY'16:7.22%, FY'17:8.41%, FY'18:9.73%, FY'19:8.02%, source: https://www.gurufocus.com/term/wacc/NAS:GOOGL/WACC-Percentage/Alphabet(Google)



from numpy import *

PE = [34.09,28.47,58.65,23.91,27.23]

WACC = [0.0751,0.0722,0.0841,0.0973,0.0802]

print(mean(PE))

print(mean(WACC))

expanded_PE = mean(PE) * (1 + 1.3)

current_target_buy_price_basic = expanded_PE * predicted_EPS.iloc[0,0]/(1+mean(WACC))

current_target_buy_price_diluted = expanded_PE * predicted_EPS.iloc[0,1]/(1+mean(WACC))

print('Basic and diluted current Target Buy Price are',current_target_buy_price_basic, 'and',current_target_buy_price_diluted,'respectively')







#Margin of Safety = (1 - 25%) * Target Buy Price

margin_of_safety_basic =  0.75 * current_target_buy_price_basic

margin_of_safety_dilluted =  0.75 * current_target_buy_price_diluted

print('Basic and diluted margin of safety are',margin_of_safety_basic,'and',margin_of_safety_dilluted,'respectively' )
#DE ratio = debt / equity

DE_ratio = google_BS.loc["FY '19","Total Liabilities"] / google_BS.loc["FY '19","Total Equity"]

print('debt=',google_BS.loc["FY '19","Total Liabilities"])

print('equity=',google_BS.loc["FY '19","Total Equity"])

print('DE ratio=',DE_ratio)





#Interest Coverage Ratio = EBIT / Interest Expense, where EBIT = Earnings before interest and taxes

#Alphabet(Google) Annual Interest Expense: FY'15:104000 FY'16:124000 FY'17:109000 FY'18:114000 FY'19:100000, source: https://www.gurufocus.com/term/InterestExpense/NAS:GOOGL/Interest-Expense/Alphabet(Google) 

#Alphabet(Google) Annual EBIT: FY'15:19755000 FY'16:14274000 FY'17:27302000 FY'18:35027000 FY'19:39725000, source: https://www.gurufocus.com/term/EBIT/NAS:GOOGL/EBIT/Alphabet(Google)



google_ICR = pd.DataFrame({'Interest_Expense':[104000,124000,109000,114000,100000],

               'EBIT':[19755000,14274000,27302000,35027000,39725000]},

               index=["FY'15","FY'16","FY'17","FY'18","FY'19"])



google_ICR["Interest_Coverage_Ratio"] = google_ICR["EBIT"]/google_ICR["Interest_Expense"]

print(google_ICR)



google_ICR["Interest_Coverage_Ratio"].plot()