# Do the below in kaggle

#!pip install plotly==4.4.1

#!pip install chart_studio

#!pip install xlrd # for reading_excel files with extensions .xlsx into a pandas dataframe
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

_, NKE_PL, NKE_BS, NKE_CF = excel_to_df("/kaggle/input/nkedata/nike.xlsx")
NKE_BS
del(NKE_BS["Assets"])

NKE_BS
NKE_BS["_Total Current Assets"] = NKE_BS["Cash, Cash Equivalents & Short Term Investments"] + NKE_BS["Accounts & Notes Receivable"] + NKE_BS["Inventories"] + NKE_BS["Other Short Term Assets"]
NKE_BS[["_Total Current Assets", "Total Current Assets"]]
NKE_BS["_NonCurrent Assets"] = NKE_BS["Property, Plant & Equipment, Net"] + NKE_BS["Other Long Term Assets"]
NKE_BS["_Total Assets"] = NKE_BS["_NonCurrent Assets"] + NKE_BS["_Total Current Assets"] 
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
import chart_studio

# chart_studio.tools.set_credentials_file(username='your_username', api_key='your_apikey') get this from 

# https://chart-studio.plot.ly/feed/#/



# Un-remark the code below and add your own your_username and own your_apikey

chart_studio.tools.set_credentials_file(username='19447973', api_key='U2vwyZprSeHsi9zBIwI6')

import chart_studio.plotly as py

import plotly.graph_objs as go

from tk_library_py import combine_regexes

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

py.plot(fig_bs, filename='Total Assets and Liabilities')

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

        x=NKE_BS.index,

        y=NKE_BS[ col ],

        name=col

    )    

    asset_data.append(asset_bar)

    

layout_assets = go.Layout(

    barmode='stack'

)



fig_bs_assets = go.Figure(data=asset_data, layout=layout)

py.plot(fig_bs_assets, filename='Total Assets Breakdown')

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

        x=NKE_BS.index,

        y=NKE_BS[ col ],

        name=col

    )    

    liability_data.append(liability_bar)

    

layout_liabilitys = go.Layout(

    barmode='stack'

)



fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout)

py.plot(fig_bs_liabilitys, filename='Total liabilities Breakdown')

fig_bs.show()
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



fig_bs_PR = go.Figure(data=PR_data, layout=layout)

py.plot(fig_bs_PR, filename='Accounts & Notes Receivable vs Payables & Accruals')

fig_bs.show()
NKE_BS["Inventories"].plot()
NKE_BS[ ["Property, Plant & Equipment, Net", "Other Long Term Assets"] ].plot()
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



fig_bs_AAA = go.Figure(data=AAA_data, layout=layout)

py.plot(fig_bs_AAA, filename='Total Long Term Assets')

fig_bs.show()
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
# NKE has no preferred stock, no intengible assets, and no goodwill



NKE_BS["book value"] = NKE_BS["Total Assets"] - NKE_BS["Total Liabilities"]

NKE_BS["book value"].plot()
NKE_BS["current ratio"] = NKE_BS["Total Current Assets"] / NKE_BS["Total Current Liabilities"]
NKE_BS["current ratio"].plot()
#Calculate 1.	Price-to-Earnings Growth Ratio (PEG forward)  

#using this formula – PEG = Price-to-Earnings Ratio/Earnings-Growth-Rate

#https://www.investopedia.com/ask/answers/012715/what-considered-good-peg-price-earnings-growth-ratio.asp



PE_RATIO = 36.21 # FROM SIMFIN WEBSITE: https://simfin.com/data/companies/121214



# FROM NASDAQ WEBSITE: https://www.nasdaq.com/symbol/xom/earnings-growth

GROWTH_RATE = 0.1 # Forcast over the five next years



PEG_ratio = PE_RATIO / (GROWTH_RATE*100)



print("NKE's PEG Ratio is", PEG_ratio)
#End of Value Investing Stock Analysis Template
#Earnings Per Share (EPS) Annual Compounded Growth Rate



#EPS data from https://www.macrotrends.net/stocks/charts/NKE/nike/eps-earnings-per-share-diluted

#CAGR = (1+Growth Rate)^(365/Days)-1, where (End Value / Start Value)=(1+Growth Rate) and (1/Years)=(365/Days)



import pandas as pd

import math



EPS = pd.read_csv('../input/findata/nikeeps.csv',names = ['year','EPS'])

Growth_Rate = EPS['year'][0] / EPS['year'][9] - 1

print('Growth_Rate: ' + str(Growth_Rate))

CAGR = math.pow((1+Growth_Rate),10)-1

print('CAGR: '+ str(CAGR))



#Estimate EPS 10 years from now

EPS_Future = EPS['EPS'][0] * pow(1+CAGR,10)

print('EPS_Future: '+ str(EPS_Future))



#Determine Current Target Buy Price

EPS = pd.read_csv('../input/findata/nikeeps.csv',names = ['year','EPS'])

EPS = EPS.astype(float)

Price = pd.read_csv('../input/findata/nikepricedata.csv',names = ['year','price'])

Price = Price.astype(float)

df=pd.DataFrame() #empty dataframe  

df['Year'] = EPS['year']

df['EPS'] = EPS['EPS']

df['Price'] = Price['price']

df['PE'] = Price['price']/EPS['EPS']

Average_of_PE = df["PE"].mean()

print('Average of PE: ' + str(Average_of_PE))

prediction = EPS_Future*Average_of_PE

print('Price ten years from now prediction: ' + str(prediction))

#WACC = 0.05 get from Finbox

a=pow(1.05,10)

print(a)

Target = prediction/a

print('Target buy price: ' + str(Target))



#Margin of Safety

Safety = Target * 0.75

print('Margin of Safety: ' + str(Safety))



#debt to equity ratio

NKE_BS['D/E ratio'] = NKE_BS['Total Liabilities']/NKE_BS['Total Equity']

NKE_BS['D/E ratio'].plot()



#Interest Coverage Ratio

#get EBIT data from: https://www.koyfin.com/company/NKE/ebit 

#get interest expense data from: https://www.koyfin.com/company/NKE/ebitda_interest_expense

EI = pd.read_csv('../input/findata/intebt.csv',names = ['year','Interest_Expense','EBIT'])

EI['Interest_Coverage_Ratio'] = EI['EBIT']/EI['Interest_Expense']

print(EI)



print(df)