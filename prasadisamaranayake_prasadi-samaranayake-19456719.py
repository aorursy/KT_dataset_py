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

_, amazon_PL, amazon_BS, amazon_CF = excel_to_df("/kaggle/input/amazon/SimFin-data.xlsx")
amazon_BS
del(amazon_BS["Assets"])

del(amazon_BS["Liabilities"])

del(amazon_BS["Preferred Equity"])
amazon_BS
amazon_BS["_Total Current Assets"] = amazon_BS["Cash, Cash Equivalents & Short Term Investments"] + amazon_BS["Accounts & Notes Receivable"] + amazon_BS["Inventories"] + amazon_BS["Other Short Term Assets"]
amazon_BS[["_Total Current Assets", "Total Current Assets"]]
amazon_BS[["Total Assets"]]
amazon_BS["_NonCurrent Assets"] = amazon_BS["Property, Plant & Equipment, Net"] + amazon_BS["Other Long Term Assets"]
amazon_BS["_Total Assets"] = amazon_BS["_NonCurrent Assets"] + amazon_BS["_Total Current Assets"] 
amazon_BS["_Total Liabilities"] = amazon_BS["Total Current Liabilities"] + amazon_BS["Total Noncurrent Liabilities"]
amazon_BS[["Total Current Liabilities"]]
amazon_BS[["_Total Liabilities", "Total Liabilities"]]
%matplotlib inline

amazon_BS[["Total Assets", "Total Liabilities", "Total Equity"]].plot()
good_stuff = '''

Cash, Cash Equivalents & Short Term Investments

Accounts & Notes Receivable

Inventories

Other Short Term Assets

'''



asset_columns = [ x for x in good_stuff.strip().split("\n") ]

asset_columns
amazon_BS[ asset_columns ].plot()
import chart_studio

# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 

# https://chart-studio.plot.ly/feed/#/



# Un-remark the code below and add your own your_username and own your_apikey

chart_studio.tools.set_credentials_file(username='Prasadi', api_key='Mn20ibferPxznVsCdXcO')
import chart_studio.plotly as py

import plotly.graph_objs as go

from tk_library_py import combine_regexes

assets = go.Bar(

    x=amazon_BS.index,

    y=amazon_BS["Total Assets"],

    name='Assets'

)

liabilities = go.Bar(

    x=amazon_BS.index,

    y=amazon_BS["Total Liabilities"],

    name='Liabilities'

)



shareholder_equity = go.Scatter(

    x=amazon_BS.index,

    y=amazon_BS["Total Equity"],

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

        x=amazon_BS.index,

        y=amazon_BS[ col ],

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

Other Short Term Liabilities

Long Term Debt

Other Long Term Liabilities

'''





for col in columns.strip().split("\n"):

    liability_bar = go.Bar(

        x=amazon_BS.index,

        y=amazon_BS[ col ],

        name=col

    )    

    liability_data.append(liability_bar)

    

layout_liabilitys = go.Layout(

    barmode='stack'

)



fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)

py.iplot(fig_bs_liabilitys, filename='Total liabilities Breakdown')
good_stuff2 = '''

Payables & Accruals

Other Short Term Liabilities 

'''
liabilities_columns = [ x for x in good_stuff2.strip().split("\n") ]

liabilities_columns
amazon_BS[ liabilities_columns ].plot()
amazon_BS["working capital"] = amazon_BS["Total Current Assets"] - amazon_BS["Total Current Liabilities"]
amazon_BS[["working capital"]].plot()
amazon_BS["Net Asset Value"] = amazon_BS["Total Current Assets"] - amazon_BS["Total Current Liabilities"] - amazon_BS["Total Noncurrent Liabilities"]
amazon_BS[["Net Asset Value"]].plot()
amazon_BS[["Accounts & Notes Receivable", "Payables & Accruals"]].plot()
# Using Chart Studio in Plotly 



PR_data = []

columns = '''

Accounts & Notes Receivable

Payables & Accruals

'''



for col in columns.strip().split("\n"):

    PR_Scatter = go.Scatter(

        x=amazon_BS.index,

        y=amazon_BS[ col ],

        name=col

    )    

    PR_data.append(PR_Scatter)

    

layout_PR = go.Layout(

    barmode='stack'

)



fig_bs_PR = go.Figure(data=PR_data, layout=layout_PR)

py.iplot(fig_bs_PR, filename='Accounts & Notes Receivable vs Payables & Accruals')
amazon_BS["Inventories"].plot()
amazon_BS[ ["Property, Plant & Equipment, Net", "Other Long Term Assets"] ].plot()
# Using Plotly



AAA_data = []

columns = '''

Property, Plant & Equipment, Net

Other Long Term Assets

'''





for col in columns.strip().split("\n"):

    AAA_bar = go.Bar(

        x=amazon_BS.index,

        y=amazon_BS[ col ],

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

'''



equity_columns = [ x for x in equity_columns.strip().split("\n")]

equity_columns
amazon_BS[ equity_columns ].plot()
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

        x=amazon_BS.index,

        y=amazon_BS[ col ],

        name=col

    )    

    equity_data.append(equity_Scatter)

    

layout_equity = go.Layout(

    barmode='stack'

)



fig_bs_equity = go.Figure(data=equity_data, layout=layout_equity)

py.iplot(fig_bs_equity, filename='Total Equity')
# Amazon has no preferred stock, no intangible assets, and no goodwill



amazon_BS["book value"] = amazon_BS["Total Assets"] - amazon_BS["Total Liabilities"]

amazon_BS["book value"].plot()
amazon_BS["current ratio"] = amazon_BS["Total Current Assets"] / amazon_BS["Total Current Liabilities"]
amazon_BS["current ratio"].plot()
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  

#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate

#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp



PE_RATIO = 88.64 # FROM SIMFIN WEBSITE: https://simfin.com/data/companies/62747



# FROM NASDAQ WEBSITE: https://finance.yahoo.com/quote/amzn/analysis/

GROWTH_RATE = 0.3007 # Forecast over the five next years



PEG_ratio = PE_RATIO / (GROWTH_RATE*100)



print("Amazon's PEG Ratio is", PEG_ratio)
#End of Value Investing Stock Analysis Template
#*****OWN CODE*****
# Show the files and their pathnames

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Add your AmazonAnnualEPS.xlsx using the '+ Add Data' top right button 



import pandas as pd

AmazonAnnualEPS = pd.read_excel("/kaggle/input/amazonannualeps/AmazonAnnualEPS.xlsx")
AmazonAnnualEPS= AmazonAnnualEPS.dropna(axis=1, how='all')

AmazonAnnualEPS

AmazonAnnualEPS= AmazonAnnualEPS.dropna(axis=0, how='any')

AmazonAnnualEPS

AmazonAnnualEPS=AmazonAnnualEPS.tail(5)

AmazonAnnualEPS

AmazonAnnualEPS=AmazonAnnualEPS.rename(columns={'Data from macrotrends.net: https://www.macrotrends.net/stocks/charts/AMZN/amazon/eps-earnings-per-share-diluted':"Year",'Unnamed: 1':'Annual EPS'})

AmazonAnnualEPS
#Create a table of Annual EPS for the past 5 years (2015-2019)

AmazonAnnualEPS1=AmazonAnnualEPS.set_index('Year', drop=True)

AmazonAnnualEPS1
#Divide the EPS of 2019 by the EPS of each year

EPS2019=AmazonAnnualEPS1.iloc[4]

EPS2019_YEAR = 23.01 / AmazonAnnualEPS1["Annual EPS"]

EPS2019_YEAR
#Calculate the Earnings Per Share (EPS) Compound Annual Growth Rate for each year using the formula ((EPS2019/EPSYearN)^(2019-N))-1

EPS2015n=((EPS2019_YEAR.iloc[0])**(1/5))-1

EPS2016n=((EPS2019_YEAR.iloc[1])**(1/4))-1

EPS2017n=((EPS2019_YEAR.iloc[2])**(1/3))-1

EPS2018n=((EPS2019_YEAR.iloc[3])**(1/2))-1

EPS2019n=((EPS2019_YEAR.iloc[4])**(1/1))-1

EPSCAGR= pd.DataFrame({'Year': [2015, 2016, 2017, 2018, 2019],'EPS CAGR': [EPS2015n,EPS2016n,EPS2017n,EPS2018n,EPS2019n]})

EPSCAGR

EPSCAGR=EPSCAGR.rename(columns={0:'Year'})

EPSCAGR

EPSCAGR=EPSCAGR.set_index('Year', drop=True)

EPSCAGR
#Calculate the average EPSCAGR of the past 5 years

EPSCAGR=EPSCAGR.iloc[:4,0]

EPSCAGR

TotalEPSCAGR=EPSCAGR.sum()

TotalEPSCAGR

AverageEPSCAGR=TotalEPSCAGR/5

AverageEPSCAGR

print("The average EPS CAGR is", AverageEPSCAGR)
#Calculate the projected EPS for the year 2029 using the formula EPS2019*(1+AverageEPSCAGR)^10

ProjectedEPS2029=23.01*((1+AverageEPSCAGR)**10)

ProjectedEPS2029

print("The estimated EPS in 2019 is", ProjectedEPS2029)
#Calculate the Current Target Buy Price using the formula EPS2019*PEratio

#PEratio=88.64 from https://simfin.com/data/companies/62747

EPS2019num=AmazonAnnualEPS1.iloc[4,0]

PEratio=88.64

CurrentTargetBuyPrice=EPS2019num*PEratio

CurrentTargetBuyPrice

print("The Current Target Buy Price is", CurrentTargetBuyPrice)
#Find the Margin of Safety, which is Current Target Buy Price +- 25%

Variation=0.25*CurrentTargetBuyPrice

Variation

MarginOfSafety=print("The Margin of Safety is","(",CurrentTargetBuyPrice-Variation,",",CurrentTargetBuyPrice+Variation,")")

MarginOfSafety
amazonLandE=amazon_BS[["Total Liabilities","Total Equity"]]

amazonLandE
#Calculate the Debt to Equity Ratio in 2019 using the formula Total Liabilities/Total Equity

DebtToEquityRatio= amazonLandE.iloc[9,0]/amazonLandE.iloc[9,1]

DebtToEquityRatio

print("The Debt to Equity Ratio is", DebtToEquityRatio)
amazonDA=amazon_CF[['Depreciation & Amortization']]

amazonDA
#Find EBIT by using the formula EBITDA + Depreciation & Amortization

#EBITDA=36,330 from https://simfin.com/data/companies/62747

EBIT=amazonDA.iloc[9,0]+36330

EBIT

print("EBIT is", EBIT)
#Calculate the Interest Coverage Ratio in 2019 using the formula EBIT/Interest Expense

#Interest Expense=1600 from https://www.marketwatch.com/investing/stock/amzn/financials

InterestExpense=1600

InterestCoverageRatio=EBIT/InterestExpense

InterestCoverageRatio

print("The Interest Coverage Ratio is", InterestCoverageRatio)